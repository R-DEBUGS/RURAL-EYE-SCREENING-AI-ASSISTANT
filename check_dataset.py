
from pathlib import Path

DATASET_DIR = Path("data/raw")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\nRURAL EYE AI - DATASET REPORT")
print("-" * 40)

for folder in DATASET_DIR.iterdir():
    if folder.is_dir():
        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]

        print(f"{folder.name}: {len(images)} images")

print("-" * 40)
print("Dataset scan completed.")
