
from pathlib import Path
import csv
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/split")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# --------------------------------------------------
# 2. IMAGE PREPROCESSING
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# 3. LOAD DATASETS
# --------------------------------------------------

train_path = DATA_DIR / "train"
val_path = DATA_DIR / "val"
test_path = DATA_DIR / "test"

for path in [train_path, val_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset folder missing: {path}")

train_data = datasets.ImageFolder(
    train_path, transform=train_transform
)
val_data = datasets.ImageFolder(
    val_path, transform=eval_transform
)
test_data = datasets.ImageFolder(
    test_path, transform=eval_transform
)

classes = train_data.classes
num_classes = len(classes)

for name, dataset in [
    ("Validation", val_data),
    ("Test", test_data),
]:
    if dataset.classes != classes:
        raise ValueError(
            f"{name} classes {dataset.classes} do not match "
            f"training classes {classes}"
        )

if num_classes < 2:
    raise ValueError("At least two classes are required.")

if any(len(ds) == 0 for ds in [train_data, val_data, test_data]):
    raise ValueError("Train, validation, and test sets must be non-empty.")

print("Classes:", classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# 4. BUILD RESNET18 WITH TRANSFER LEARNING
# --------------------------------------------------

# Pretrained ImageNet weights; the final layer is replaced
# to match our eye-disease classes.
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------
# 5. TRAIN AND VALIDATE
# --------------------------------------------------

history = []
best_val_accuracy = -1.0
best_state = None

for epoch in range(EPOCHS):
    start_time = time.time()

    # Training phase
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # Validation phase
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Time: {time.time() - start_time:.1f}s"
    )

    # Save the checkpoint with the best validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "classes": classes,
            "architecture": "ResNet18",
            "epoch": epoch + 1,
            "val_accuracy": val_accuracy,
        }, MODEL_PATH)

        print("  Best validation checkpoint saved.")

# --------------------------------------------------
# 6. SAVE TRAINING HISTORY AND GRAPHS
# --------------------------------------------------

history_path = RESULTS_DIR / "resnet18_training_history.csv"

with open(history_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

epochs = [item["epoch"] for item in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    epochs, [item["train_loss"] for item in history],
    label="Training Loss"
)
axes[0].plot(
    epochs, [item["val_loss"] for item in history],
    label="Validation Loss"
)
axes[0].set_title("ResNet18 Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    epochs, [item["train_accuracy"] * 100 for item in history],
    label="Training Accuracy"
)
axes[1].plot(
    epochs, [item["val_accuracy"] * 100 for item in history],
    label="Validation Accuracy"
)
axes[1].set_title("ResNet18 Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "resnet18_training_curves.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# 7. FINAL EVALUATION ON HELD-OUT TEST SET
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        actual.extend(labels.numpy().tolist())
        predicted.extend(
            outputs.argmax(dim=1).cpu().numpy().tolist()
        )

test_accuracy = accuracy_score(actual, predicted)
macro_f1 = f1_score(
    actual, predicted, average="macro", zero_division=0
)
macro_precision = __import__(
    "sklearn.metrics", fromlist=["precision_score"]
).precision_score(
    actual, predicted, average="macro", zero_division=0
)
macro_recall = __import__(
    "sklearn.metrics", fromlist=["recall_score"]
).recall_score(
    actual, predicted, average="macro", zero_division=0
)

print("\n========== RESNET18 TEST RESULTS ==========")
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    labels=list(range(num_classes)),
    target_names=classes,
    digits=4,
    zero_division=0,
))

matrix = confusion_matrix(
    actual, predicted, labels=list(range(num_classes))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=classes,
)
display.plot(cmap="Blues", values_format="d")
plt.title("ResNet18 Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "resnet18_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

with open(
    RESULTS_DIR / "resnet18_test_metrics.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Architecture", "ResNet18"])
    writer.writerow(["Best Epoch", checkpoint["epoch"]])
    writer.writerow(["Test Accuracy", test_accuracy])
    writer.writerow(["Macro Precision", macro_precision])
    writer.writerow(["Macro Recall", macro_recall])
    writer.writerow(["Macro F1-score", macro_f1])

print("\nSaved model:", MODEL_PATH)
print("Saved training history:", history_path)
print("Saved training curves:",
      RESULTS_DIR / "resnet18_training_curves.png")
print("Saved test metrics:",
      RESULTS_DIR / "resnet18_test_metrics.csv")
print("Saved confusion matrix:",
      RESULTS_DIR / "resnet18_confusion_matrix.png")
