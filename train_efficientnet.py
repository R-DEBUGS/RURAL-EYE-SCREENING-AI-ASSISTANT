import os
import json
import copy
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader


# ---------------- CONFIGURATION ----------------

DATA_DIR = "data/split"
MODEL_DIR = "models"
RESULTS_DIR = "results"

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
SEED = 42

MODEL_FILENAME = "efficientnet_b0_best.pth"
HISTORY_FILENAME = "efficientnet_b0_training_history.json"
LOSS_GRAPH_FILENAME = "efficientnet_b0_training_loss.png"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- DATASET PATHS ----------------

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

for folder in [train_dir, val_dir, test_dir]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Required dataset folder not found: {folder}"
        )


# ---------------- TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------- DATASETS AND LOADERS ----------------

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=eval_transform)
test_data = datasets.ImageFolder(test_dir, transform=eval_transform)

class_names = train_data.classes
num_classes = len(class_names)

if num_classes < 2:
    raise ValueError("At least two class folders are required.")

if val_data.classes != class_names or test_data.classes != class_names:
    raise ValueError(
        "Train, validation, and test folders must contain the same "
        "class folders with matching names."
    )

print("Classes:", class_names)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))
print("Note: the test set is reserved for final evaluation and is not used here.")

with open(os.path.join(MODEL_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# ---------------- CREATE EFFICIENTNET-B0 ----------------

weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------- TRAIN AND VALIDATE ----------------

train_losses = []
val_losses = []
val_accuracies = []

best_val_accuracy = -1.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    epoch_start = time.perf_counter()

    model.train()
    total_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)

    epoch_train_loss = total_train_loss / len(train_data)
    train_losses.append(epoch_train_loss)

    model.eval()
    total_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_val_loss = total_val_loss / len(val_data)
    val_accuracy = 100.0 * correct / total

    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_accuracy)

    elapsed = time.perf_counter() - epoch_start
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}% | "
        f"Time: {elapsed:.1f}s"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(
            best_weights,
            os.path.join(MODEL_DIR, MODEL_FILENAME)
        )
        print("  Saved new best validation checkpoint.")


# ---------------- SAVE HISTORY AND CLASS METADATA ----------------

history = {
    "model": "EfficientNet-B0",
    "epochs": list(range(1, EPOCHS + 1)),
    "training_loss": train_losses,
    "validation_loss": val_losses,
    "validation_accuracy": val_accuracies,
    "class_names": class_names,
    "best_validation_accuracy_percent": best_val_accuracy,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "seed": SEED
}

with open(os.path.join(RESULTS_DIR, HISTORY_FILENAME), "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)


# ---------------- PLOT TRAINING / VALIDATION LOSS ----------------

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_losses, marker="o", label="Training Loss")
plt.plot(epochs_range, val_losses, marker="s", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("EfficientNet-B0 Training and Validation Loss")
plt.xticks(list(epochs_range))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

graph_path = os.path.join(RESULTS_DIR, LOSS_GRAPH_FILENAME)
plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nEfficientNet-B0 training complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Checkpoint: models/{MODEL_FILENAME}")
print(f"Loss graph: results/{LOSS_GRAPH_FILENAME}")
print(f"Training history: results/{HISTORY_FILENAME}")
