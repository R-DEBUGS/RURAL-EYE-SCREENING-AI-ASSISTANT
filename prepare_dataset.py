
from pathlib import Path
import random
import shutil
from collections import Counter

# Original dataset and output locations
SOURCE = Path("data/raw")
OUTPUT = Path("data/split")

CLASSES = ["cataract", "normal", "pterygium"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

# Find images recursively, including nested folders
images_by_class = {}

for class_name in CLASSES:
    folder = SOURCE / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if len(images) < 10:
        raise ValueError(
            f"Only {len(images)} images found for {class_name}. "
            "Check your folder structure."
        )

    images_by_class[class_name] = images

# Avoid accidentally overwriting an existing split
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(
        f"{OUTPUT} already contains files. "
        "Rename or back up that folder before running again."
    )

# Split each class: 70% train, 15% validation, 15% test
for class_name, images in images_by_class.items():
    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in splits.items():
        destination = OUTPUT / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(split_images):
            # Unique names avoid collisions between nested folders
            new_name = f"{index:05d}_{image_path.name}"
            shutil.copy2(image_path, destination / new_name)

    print(
        f"{class_name}: "
        f"{len(splits['train'])} train, "
        f"{len(splits['val'])} validation, "
        f"{len(splits['test'])} test"
    )

print("\nDataset split completed.")
print(f"Output folder: {OUTPUT.resolve()}")
