import os
import json
import time
import csv

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = "data/split"
MODEL_DIR = "models"
RESULTS_DIR = "results"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "efficientnet_b0_best.pth")
IMAGE_SIZE = 224
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open(os.path.join(MODEL_DIR, "class_names.json"), "r", encoding="utf-8") as f:
    class_names = json.load(f)
num_classes = len(class_names)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dir = os.path.join(DATA_DIR, "test")
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

if test_data.classes != class_names:
    raise ValueError(
        f"Class order mismatch. Checkpoint metadata: {class_names}; "
        f"test folders: {test_data.classes}"
    )

test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

# Recreate the exact EfficientNet-B0 architecture with a classifier for this dataset.
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

state = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []
batch_inference_times = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels_device = labels.to(device)

        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        batch_inference_times.append(elapsed / images.size(0) * 1000.0)
        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

y_true = np.asarray(all_labels)
y_pred = np.asarray(all_predictions)
y_prob = np.asarray(all_probabilities)

accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

try:
    macro_auc = roc_auc_score(
        y_true, y_prob, multi_class="ovr", average="macro",
        labels=list(range(num_classes))
    )
except ValueError as exc:
    macro_auc = None
    print("Could not calculate macro ROC-AUC:", exc)

checkpoint_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
mean_inference_ms = float(np.mean(batch_inference_times))

results = {
    "model": "EfficientNet-B0",
    "test_images": len(test_data),
    "correct_predictions": int((y_true == y_pred).sum()),
    "test_accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "macro_roc_auc_ovr": None if macro_auc is None else float(macro_auc),
    "checkpoint_size_mb": float(checkpoint_mb),
    "mean_inference_ms_per_image": mean_inference_ms,
    "training_epochs_completed": 5,
    "best_validation_accuracy_percent": 95.21,
    "class_names": class_names,
}

print("\n=== EfficientNet-B0 TEST RESULTS ===")
for key, value in results.items():
    if key != "class_names":
        print(f"{key}: {value}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4, zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("Confusion matrix (rows=true, columns=predicted):")
print(cm)

with open(os.path.join(RESULTS_DIR, "efficientnet_b0_test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(RESULTS_DIR, "efficientnet_b0_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    ))
    f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
    f.write(np.array2string(cm))

with open(os.path.join(RESULTS_DIR, "efficientnet_b0_test_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[k for k in results.keys() if k != "class_names"])
    writer.writeheader()
    writer.writerow({k: v for k, v in results.items() if k != "class_names"})

print("\nSaved:")
print("results/efficientnet_b0_test_metrics.json")
print("results/efficientnet_b0_test_metrics.csv")
print("results/efficientnet_b0_classification_report.txt")
print("\nImportant: this is the first test evaluation for this checkpoint. Do not use test results to tune the model.")
