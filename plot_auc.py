
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 1. Configuration
DATA_DIR = Path("data/split/test")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "Original": Path("models/eye_classifier.pth"),
    "Improved": Path("models/eye_classifier_improved.pth"),
}

IMAGE_SIZE = 224
BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Load test dataset with the same preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {DATA_DIR}")

test_data = datasets.ImageFolder(
    DATA_DIR, transform=transform
)
classes = test_data.classes
num_classes = len(classes)

if num_classes < 2:
    raise ValueError("ROC-AUC requires at least two classes.")

loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Device:", device)
print("Classes:", classes)
print("Test images:", len(test_data))


# 3. Load MobileNetV3 checkpoint
def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    saved_classes = (
        checkpoint.get("classes")
        if isinstance(checkpoint, dict)
        else None
    )

    if saved_classes is not None and list(saved_classes) != classes:
        raise ValueError(
            f"Class mismatch: checkpoint={saved_classes}, "
            f"test dataset={classes}"
        )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, num_classes
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# 4. Get prediction probabilities
def get_probabilities(model):
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


# 5. Calculate AUC and plot ROC curves
fig, axes = plt.subplots(
    1, len(MODEL_PATHS), figsize=(8 * len(MODEL_PATHS), 6),
    squeeze=False
)
axes = axes.ravel()

all_results = {}

for ax, (model_name, model_path) in zip(axes, MODEL_PATHS.items()):
    print(f"\nEvaluating {model_name} model...")

    model = load_model(model_path)
    y_true, y_probs = get_probabilities(model)
    del model

    # One-vs-rest ROC for each class
    y_binary = label_binarize(
        y_true, classes=list(range(num_classes))
    )

    class_aucs = {}

    for i, class_name in enumerate(classes):
        if len(np.unique(y_binary[:, i])) < 2:
            print(f"Skipping {class_name}: test set needs positive "
                  "and negative examples.")
            continue

        fpr, tpr, _ = roc_curve(y_binary[:, i], y_probs[:, i])
        class_auc = auc(fpr, tpr)
        class_aucs[class_name] = class_auc

        ax.plot(
            fpr, tpr,
            label=f"{class_name} (AUC = {class_auc:.3f})"
        )

        print(f"{class_name} AUC: {class_auc:.4f}")

    if not class_aucs:
        raise ValueError("Could not calculate AUC for any class.")

    # Macro-average AUC across classes
    macro_auc = roc_auc_score(
        y_true, y_probs, labels=list(range(num_classes)),
        multi_class="ovr", average="macro"
    )
    all_results[model_name] = {
        "class_aucs": class_aucs,
        "macro_auc": macro_auc,
    }

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_title(f"{model_name} Model (Macro AUC = {macro_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    print(f"{model_name} Macro AUC: {macro_auc:.4f}")

fig.suptitle("ROC-AUC Comparison: MobileNetV3 Models")
fig.tight_layout()

output_path = RESULTS_DIR / "roc_auc_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# 6. Save AUC values to CSV
import csv

csv_path = RESULTS_DIR / "roc_auc_scores.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Class", "AUC"])

    for model_name, result in all_results.items():
        for class_name, score in result["class_aucs"].items():
            writer.writerow([model_name, class_name, score])

        writer.writerow([
            model_name, "Macro Average", result["macro_auc"]
        ])

print("\nROC graph saved to:", output_path)
print("AUC scores saved to:", csv_path)
