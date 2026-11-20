
from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_DIR = Path("data/raw")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\nRURAL EYE AI - DATASET INSPECTION")
print("=" * 40)

for folder in sorted(DATASET_DIR.iterdir()):
    if not folder.is_dir():
        continue

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    formats = Counter()
    dimensions = Counter()
    unreadable = 0

    for path in images:
        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                formats[img.format or "Unknown"] += 1
                dimensions[f"{img.width}x{img.height}"] += 1

        except Exception:
            unreadable += 1

    print(f"\nClass: {folder.name}")
    print(f"Files found: {len(images)}")
    print(f"Readable images: {len(images) - unreadable}")
    print(f"Unreadable images: {unreadable}")
    print(f"Formats: {dict(formats)}")
    print(f"Common dimensions: {dimensions.most_common(5)}")

print("\nInspection completed.")
