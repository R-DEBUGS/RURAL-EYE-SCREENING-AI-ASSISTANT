import os, json, time, csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize

DATA_DIR = 'data/split'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
IMAGE_SIZE = 224
BATCH_SIZE = 16
os.makedirs(RESULTS_DIR, exist_ok=True)

# Checkpoint paths and model architecture definitions
MODEL_CONFIGS = [
    ('MobileNetV3-Small', os.path.join(MODEL_DIR, 'eye_classifier.pth'), 'mobilenet'),
    ('ResNet18', os.path.join(MODEL_DIR, 'resnet18_best.pth'), 'resnet'),
    ('EfficientNet-B0', os.path.join(MODEL_DIR, 'efficientnet_b0_best.pth'), 'efficientnet'),
]

def load_state(path, device):
    # Supports plain state_dict checkpoints and common wrapped checkpoints.
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if isinstance(obj, dict) and any(k.startswith('module.') for k in obj):
        obj = {k.replace('module.', '', 1): v for k, v in obj.items()}
    return obj

def build_model(kind, nclasses):
    if kind == 'mobilenet':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, nclasses)
    elif kind == 'resnet':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, nclasses)
    elif kind == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, nclasses)
    else:
        raise ValueError(kind)
    return model

with open(os.path.join(MODEL_DIR, 'class_names.json'), encoding='utf-8') as f:
    class_names = json.load(f)
num_classes = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
if test_data.classes != class_names:
    raise ValueError(f'Class order mismatch: metadata={class_names}, test folders={test_data.classes}')
loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
y_true = np.array(test_data.targets)

all_results = []
all_predictions = {}
all_probabilities = {}
all_cms = {}

for name, path, kind in MODEL_CONFIGS:
    print(f'\nEvaluating {name}...')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    model = build_model(kind, num_classes)
    state = load_state(path, device)
    model.load_state_dict(state)
    model.to(device).eval()
    preds, probs = [], []
    timed_images = 0
    inference_seconds = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == 'cuda': torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            timed_images += images.size(0)
            p = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(p, dim=1).cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())
    pred = np.asarray(preds)
    prob = np.asarray(probs)
    if len(pred) != len(y_true):
        raise RuntimeError(f'{name}: prediction count {len(pred)} does not match labels {len(y_true)}')
    cm = confusion_matrix(y_true, pred, labels=list(range(num_classes)))
    try:
        macro_auc = roc_auc_score(y_true, prob, multi_class='ovr', average='macro', labels=list(range(num_classes)))
    except ValueError:
        macro_auc = float('nan')
    row = {
        'model': name,
        'test_images': int(len(y_true)),
        'test_accuracy_percent': 100 * accuracy_score(y_true, pred),
        'macro_precision_percent': 100 * precision_score(y_true, pred, average='macro', zero_division=0),
        'macro_recall_percent': 100 * recall_score(y_true, pred, average='macro', zero_division=0),
        'macro_f1_percent': 100 * f1_score(y_true, pred, average='macro', zero_division=0),
        'weighted_f1_percent': 100 * f1_score(y_true, pred, average='weighted', zero_division=0),
        'macro_roc_auc_ovr': float(macro_auc),
        'checkpoint_size_mb': os.path.getsize(path) / (1024 * 1024),
        'mean_inference_ms_per_image': inference_seconds / max(timed_images, 1) * 1000,
    }
    all_results.append(row)
    all_predictions[name], all_probabilities[name], all_cms[name] = pred, prob, cm
    print(f"  Accuracy: {row['test_accuracy_percent']:.2f}% | Macro F1: {row['macro_f1_percent']:.2f}% | AUC: {row['macro_roc_auc_ovr']:.4f}")
    del model

# Save metric table CSV/JSON
with open(os.path.join(RESULTS_DIR, 'three_model_comparison.csv'), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
    writer.writeheader(); writer.writerows(all_results)
with open(os.path.join(RESULTS_DIR, 'three_model_comparison.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2)

# 1) Performance metric comparison chart
metrics = ['test_accuracy_percent', 'macro_precision_percent', 'macro_recall_percent', 'macro_f1_percent']
labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
x = np.arange(len(metrics)); width = 0.24
fig, ax = plt.subplots(figsize=(10, 6))
for i, r in enumerate(all_results):
    vals = [r[m] for m in metrics]
    ax.bar(x + (i - 1) * width, vals, width, label=r['model'])
ax.set_ylabel('Score (%)'); ax.set_title('Three-Model Performance Comparison')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 105); ax.legend(); ax.grid(axis='y', alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'three_model_performance_comparison.png'), dpi=300); plt.close(fig)

# 2) Multiclass macro ROC curve: micro-averaged curve for each model; class-specific curves in a separate figure.
y_bin = label_binarize(y_true, classes=list(range(num_classes)))
fig, ax = plt.subplots(figsize=(8, 7))
for name, prob in all_probabilities.items():
    fpr, tpr, _ = roc_curve(y_bin.ravel(), prob.ravel())
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (micro-AUC={roc_auc:.4f})')
ax.plot([0, 1], [0, 1], linestyle='--', label='Chance')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('ROC Curve Comparison (Micro-average, One-vs-Rest)')
ax.legend(loc='lower right'); ax.grid(alpha=0.25); fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'three_model_roc_comparison.png'), dpi=300); plt.close(fig)

# 3) Per-class one-vs-rest ROC curves for each model (nine curves total)
fig, ax = plt.subplots(figsize=(9, 8))
for name, prob in all_probabilities.items():
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], prob[:, idx])
        ax.plot(fpr, tpr, label=f'{name} — {class_name} (AUC={auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', label='Chance')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('Class-wise ROC Curves (One-vs-Rest)')
ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=0.25); fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'three_model_classwise_roc.png'), dpi=300); plt.close(fig)

# 4) Confusion matrices, one panel per model
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (name, cm) in zip(axes, all_cms.items()):
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(name); ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names, rotation=35, ha='right')
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > threshold else 'black')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Number of images')
fig.suptitle('Confusion Matrix Comparison'); fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.45)
fig.savefig(os.path.join(RESULTS_DIR, 'three_model_confusion_matrices.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

# 5) Efficiency comparison charts
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
names = [r['model'] for r in all_results]
axes[0].bar(names, [r['checkpoint_size_mb'] for r in all_results]); axes[0].set_title('Checkpoint Size'); axes[0].set_ylabel('MB')
axes[1].bar(names, [r['mean_inference_ms_per_image'] for r in all_results]); axes[1].set_title('Mean Inference Time'); axes[1].set_ylabel('Milliseconds per image')
for ax in axes:
    ax.tick_params(axis='x', rotation=20); ax.grid(axis='y', alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'three_model_efficiency_comparison.png'), dpi=300); plt.close(fig)

print('\nSaved comparison charts and table in results/:')
for filename in [
    'three_model_comparison.csv', 'three_model_comparison.json',
    'three_model_performance_comparison.png', 'three_model_roc_comparison.png',
    'three_model_classwise_roc.png', 'three_model_confusion_matrices.png',
    'three_model_efficiency_comparison.png'
]:
    print(' - results/' + filename)
print('\nNote: ROC curves are calculated from the current checkpoints and the same test split. Keep test set for final evaluation; do not tune models against these test results.')
