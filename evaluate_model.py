
from pathlib import Path
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    auc,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path("data/split")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

MODEL_CONFIGS = {
    "MobileNetV3-Small": {
        "path": MODEL_DIR / "eye_classifier_improved.pth",
        "architecture": "mobilenet",
    },
    "ResNet18": {
        "path": MODEL_DIR / "resnet18_best.pth",
        "architecture": "resnet",
    },
    "EfficientNet-B0": {
        "path": MODEL_DIR / "efficientnet_b0_best.pth",
        "architecture": "efficientnet",
    },
    "ShuffleNetV2": {
        "path": MODEL_DIR / "eye_classifier.pth",
        "architecture": "shufflenet",
    },
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FOUR-MODEL EYE CLASSIFICATION EVALUATION")
print("=" * 70)
print("Device:", device)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

test_path = DATA_DIR / "test"

if not test_path.is_dir():
    raise FileNotFoundError(
        f"Test dataset not found: {test_path}"
    )

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_data = datasets.ImageFolder(
    test_path,
    transform=eval_transform,
)

classes = test_data.classes
num_classes = len(classes)

if num_classes < 2:
    raise ValueError("At least two classes are required.")

loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print("Classes:", classes)
print("Test images:", len(test_data))


# ============================================================
# 3. LOAD MODEL ARCHITECTURES AND CHECKPOINTS
# ============================================================

def load_model(model_path, architecture):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format: {model_path}"
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    saved_classes = checkpoint.get("classes")

    if saved_classes is not None and list(saved_classes) != classes:
        raise ValueError(
            f"Class mismatch in {model_path}.\n"
            f"Checkpoint classes: {saved_classes}\n"
            f"Test classes: {classes}"
        )

    if architecture == "mobilenet":
        model = models.mobilenet_v3_small(weights=None)

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes,
        )

    elif architecture == "resnet":
        model = models.resnet18(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif architecture == "efficientnet":
        model = models.efficientnet_b0(weights=None)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    elif architecture == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 4. EVALUATE A MODEL
# ============================================================

def evaluate_model(model):
    actual = []
    predicted = []
    probabilities = []

    # Warm-up before timing.
    with torch.inference_mode():
        warmup = torch.zeros(
            min(BATCH_SIZE, len(test_data)),
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device,
        )

        model(warmup)

        if device.type == "cuda":
            torch.cuda.synchronize()

    inference_seconds = 0.0
    num_images = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_seconds += time.perf_counter() - start

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            actual.extend(labels.numpy().tolist())
            predicted.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

            num_images += images.size(0)

    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)
    y_probs = np.asarray(probabilities)

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    # Multiclass ROC-AUC: one-vs-rest.
    class_aucs = {}

    for i, class_name in enumerate(classes):
        binary_true = (y_true == i).astype(int)

        if len(np.unique(binary_true)) < 2:
            class_aucs[class_name] = np.nan
        else:
            class_aucs[class_name] = roc_auc_score(
                binary_true,
                y_probs[:, i],
            )

    if all(np.isfinite(v) for v in class_aucs.values()):
        macro_auc = roc_auc_score(
            y_true,
            y_probs,
            labels=list(range(num_classes)),
            multi_class="ovr",
            average="macro",
        )
    else:
        macro_auc = np.nan

    return {
        "actual": y_true,
        "predicted": y_pred,
        "probabilities": y_probs,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "class_aucs": class_aucs,
        "report": report,
        "confusion_matrix": matrix,
        "inference_ms_per_image": (
            inference_seconds / num_images * 1000
        ),
    }


# ============================================================
# 5. EVALUATE ALL FOUR MODELS
# ============================================================

results = {}
model_sizes = {}

for model_name, config in MODEL_CONFIGS.items():
    model_path = config["path"]

    print("\n" + "=" * 70)
    print("Evaluating:", model_name)
    print("Checkpoint:", model_path)

    model = load_model(
        model_path,
        config["architecture"],
    )

    result = evaluate_model(model)
    results[model_name] = result

    model_sizes[model_name] = (
        model_path.stat().st_size / (1024 * 1024)
    )

    print(f"Test accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Macro precision: {result['macro_precision']:.4f}")
    print(f"Macro recall: {result['macro_recall']:.4f}")
    print(f"Macro F1-score: {result['macro_f1']:.4f}")
    print(f"Weighted F1-score: {result['weighted_f1']:.4f}")

    if np.isfinite(result["macro_auc"]):
        print(f"Macro ROC-AUC: {result['macro_auc']:.4f}")
    else:
        print("Macro ROC-AUC: unavailable")

    print(
        f"Inference time: "
        f"{result['inference_ms_per_image']:.3f} ms/image"
    )
    print(f"Checkpoint size: {model_sizes[model_name]:.2f} MB")

    print("\nClassification report:")
    print(
        classification_report(
            result["actual"],
            result["predicted"],
            labels=list(range(num_classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(result["confusion_matrix"])

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 6. SAVE OVERALL COMPARISON TABLE
# ============================================================

metrics_path = RESULTS_DIR / "architecture_comparison.csv"

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Macro ROC-AUC",
        "Checkpoint Size MB",
        "Inference ms per image",
    ])

    for name, result in results.items():
        writer.writerow([
            name,
            result["accuracy"],
            result["macro_precision"],
            result["macro_recall"],
            result["macro_f1"],
            result["weighted_f1"],
            result["macro_auc"],
            model_sizes[name],
            result["inference_ms_per_image"],
        ])

print("\nSaved comparison table:", metrics_path)


# ============================================================
# 7. SAVE PER-CLASS METRICS
# ============================================================

class_metrics_path = (
    RESULTS_DIR / "architecture_per_class_metrics.csv"
)

with open(
    class_metrics_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Class",
        "Precision",
        "Recall",
        "F1-score",
        "Support",
        "ROC-AUC",
    ])

    for model_name, result in results.items():
        for class_name in classes:
            report = result["report"][class_name]

            writer.writerow([
                model_name,
                class_name,
                report["precision"],
                report["recall"],
                report["f1-score"],
                report["support"],
                result["class_aucs"][class_name],
            ])

print("Saved per-class metrics:", class_metrics_path)


# ============================================================
# 8. CLASS-WISE ROC CURVES
#    Cataract, Normal, and Pterygium
# ============================================================

# Match actual folder names without assuming that "Normal"
# and "Normal_Eye" are spelled identically.
class_aliases = {
    "Cataract": ["cataract"],
    "Normal": ["normal", "normal_eye", "normal eye"],
    "Pterygium": ["pterygium"],
}

normalized_classes = {
    name.lower().replace("-", "_").replace(" ", "_"): i
    for i, name in enumerate(classes)
}

class_indices = {}

for requested_name, aliases in class_aliases.items():
    matched_idx = None

    for alias in aliases:
        key = alias.lower().replace("-", "_").replace(" ", "_")

        if key in normalized_classes:
            matched_idx = normalized_classes[key]
            break

    if matched_idx is None:
        raise ValueError(
            f"Cannot find class '{requested_name}'. "
            f"Actual dataset classes: {classes}"
        )

    class_indices[requested_name] = matched_idx


# One figure with three panels and four model curves per panel.
fig, axes = plt.subplots(
    1,
    3,
    figsize=(19, 6),
)

for ax, class_name in zip(
    axes,
    ["Cataract", "Normal", "Pterygium"],
):
    class_idx = class_indices[class_name]

    for model_name, result in results.items():
        y_true = (
            result["actual"] == class_idx
        ).astype(int)

        y_score = result["probabilities"][:, class_idx]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        class_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {class_auc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1,
        label="Random classifier",
    )

    ax.set_title(f"{class_name} — One-vs-Rest ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    "Class-Wise ROC Curve Comparison of Four Models",
    fontsize=15,
)
fig.tight_layout()

roc_path = RESULTS_DIR / "classwise_roc_comparison.png"
fig.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved combined class-wise ROC curves:", roc_path)


# ============================================================
# 9. SAVE INDIVIDUAL ROC CURVES FOR EACH CLASS
# ============================================================

for class_name in ["Cataract", "Normal", "Pterygium"]:
    class_idx = class_indices[class_name]

    plt.figure(figsize=(7, 6))

    for model_name, result in results.items():
        y_true = (
            result["actual"] == class_idx
        ).astype(int)

        y_score = result["probabilities"][:, class_idx]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        class_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {class_auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random classifier",
    )

    plt.title(f"{class_name} Class-Wise ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    output_path = RESULTS_DIR / (
        f"classwise_roc_{class_name.lower()}.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved individual ROC curve:", output_path)


# ============================================================
# 10. PLOT CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    len(results),
    figsize=(6 * len(results), 5),
    squeeze=False,
)

for ax, (name, result) in zip(
    axes.ravel(),
    results.items(),
):
    display = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"],
        display_labels=classes,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"{name}\nAccuracy = {result['accuracy'] * 100:.2f}%"
    )

fig.tight_layout()

cm_path = RESULTS_DIR / "architecture_confusion_matrices.png"
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved confusion matrices:", cm_path)


# ============================================================
# 11. FOUR-MODEL PERFORMANCE COMPARISON BAR GRAPH
# ============================================================

model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.19

metric_series = [
    (
        "Accuracy",
        [results[n]["accuracy"] * 100 for n in model_names],
        -1.5 * width,
    ),
    (
        "Macro Precision",
        [results[n]["macro_precision"] * 100 for n in model_names],
        -0.5 * width,
    ),
    (
        "Macro Recall",
        [results[n]["macro_recall"] * 100 for n in model_names],
        0.5 * width,
    ),
    (
        "Macro F1",
        [results[n]["macro_f1"] * 100 for n in model_names],
        1.5 * width,
    ),
]

fig, ax = plt.subplots(figsize=(12, 6))

for label, values, offset in metric_series:
    ax.bar(
        x + offset,
        values,
        width,
        label=label,
    )

ax.set_ylabel("Score (%)")
ax.set_title("Four-Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=10, ha="right")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()

score_path = RESULTS_DIR / "architecture_performance.png"
fig.savefig(score_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved performance graph:", score_path)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("FINAL FOUR-MODEL COMPARISON")
print("=" * 85)

print(
    f"{'Model':<22}"
    f"{'Accuracy':>11}"
    f"{'Macro F1':>12}"
    f"{'Macro AUC':>12}"
    f"{'Size MB':>10}"
    f"{'Time ms':>11}"
)

for name, result in results.items():
    auc_text = (
        f"{result['macro_auc']:.4f}"
        if np.isfinite(result["macro_auc"])
        else "N/A"
    )

    print(
        f"{name:<22}"
        f"{result['accuracy']:>11.4f}"
        f"{result['macro_f1']:>12.4f}"
        f"{auc_text:>12}"
        f"{model_sizes[name]:>10.2f}"
        f"{result['inference_ms_per_image']:>11.3f}"
    )

print("\nEvaluation completed.")
print("All files saved in:", RESULTS_DIR.resolve())

print(
    "\nImportant: All models must be evaluated on the same held-out "
    "test set. Confirm that test images were not used for training "
    "or model selection before reporting the results in your paper."
)
