from pathlib import Path
import zipfile
import shutil
import random

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZIP_FILE = PROJECT_ROOT / "dataset" / "data.zip"
SOURCE_IMAGES = PROJECT_ROOT / "output" / "training_frames"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Temporary extraction directory
EXTRACT_DIR = DATASET_DIR / "_cvat_export"

# Final dataset directories
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"

TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_LABELS = DATASET_DIR / "labels" / "val"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ============================================================
# 1. CHECK INPUTS
# ============================================================

if not ZIP_FILE.exists():
    raise FileNotFoundError(f"CVAT ZIP not found: {ZIP_FILE}")

if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(
        f"Training frames folder not found: {SOURCE_IMAGES}"
    )


# ============================================================
# 2. EXTRACT CVAT ZIP
# ============================================================

print("Extracting CVAT annotations...")

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

EXTRACT_DIR.mkdir(parents=True)

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)


# ============================================================
# 3. FIND LABEL FILES
# ============================================================

label_files = list(EXTRACT_DIR.rglob("*.txt"))

# Ignore train.txt if CVAT exported it
label_files = [
    f for f in label_files
    if f.name != "train.txt"
]

if not label_files:
    raise RuntimeError("No YOLO annotation .txt files found.")

print(f"Found {len(label_files)} annotation files.")


# ============================================================
# 4. CREATE IMAGE LOOKUP
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

source_images = {
    img.name: img
    for img in SOURCE_IMAGES.iterdir()
    if img.is_file() and img.suffix.lower() in image_extensions
}

print(f"Found {len(source_images)} source images.")


# ============================================================
# 5. MATCH IMAGES WITH ANNOTATIONS
# ============================================================

pairs = []
missing_images = []

for label_file in label_files:

    image_name = label_file.stem

    matching_image = None

    for ext in image_extensions:
        candidate = source_images.get(image_name + ext)
        if candidate:
            matching_image = candidate
            break

    if matching_image:
        pairs.append((matching_image, label_file))
    else:
        missing_images.append(label_file.name)


print(f"Matched image/label pairs: {len(pairs)}")

if missing_images:
    print("\nWARNING: These annotation files have no matching image:")
    for name in missing_images:
        print("  ", name)


if not pairs:
    raise RuntimeError(
        "No image/annotation pairs could be matched."
    )


# ============================================================
# 6. SHUFFLE AND SPLIT
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(pairs)

train_count = int(len(pairs) * TRAIN_RATIO)

train_pairs = pairs[:train_count]
val_pairs = pairs[train_count:]

print()
print(f"Training images:   {len(train_pairs)}")
print(f"Validation images: {len(val_pairs)}")


# ============================================================
# 7. CREATE DIRECTORIES
# ============================================================

for directory in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 8. COPY FILES
# ============================================================

def copy_pairs(pair_list, image_destination, label_destination):

    for image_file, label_file in pair_list:

        shutil.copy2(
            image_file,
            image_destination / image_file.name
        )

        shutil.copy2(
            label_file,
            label_destination / (image_file.stem + ".txt")
        )


copy_pairs(
    train_pairs,
    TRAIN_IMAGES,
    TRAIN_LABELS
)

copy_pairs(
    val_pairs,
    VAL_IMAGES,
    VAL_LABELS
)


# ============================================================
# 9. CREATE data.yaml
# ============================================================

yaml_content = """path: .

train: images/train
val: images/val

names:
  0: object
"""

yaml_file = DATASET_DIR / "data.yaml"

yaml_file.write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

print("\nChecking dataset...")

errors = []

for image_dir, label_dir in [
    (TRAIN_IMAGES, TRAIN_LABELS),
    (VAL_IMAGES, VAL_LABELS),
]:

    for image_file in image_dir.iterdir():

        if not image_file.is_file():
            continue

        label_file = label_dir / f"{image_file.stem}.txt"

        if not label_file.exists():
            errors.append(
                f"Missing label: {label_file}"
            )


# Check YOLO annotation format
for label_dir in [TRAIN_LABELS, VAL_LABELS]:

    for label_file in label_dir.glob("*.txt"):

        lines = label_file.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(lines, 1):

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"Invalid annotation: "
                    f"{label_file.name}, line {line_number}"
                )
                continue

            try:
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]

                if class_id != 0:
                    errors.append(
                        f"Unexpected class ID in "
                        f"{label_file.name}: {class_id}"
                    )

                if not all(0 <= x <= 1 for x in values):
                    errors.append(
                        f"Coordinates outside 0-1 range in "
                        f"{label_file.name}"
                    )

            except ValueError:
                errors.append(
                    f"Invalid numeric values in "
                    f"{label_file.name}, line {line_number}"
                )


# ============================================================
# RESULT
# ============================================================

print()

if errors:

    print("DATASET CREATED WITH WARNINGS/ERRORS:")
    for error in errors:
        print("  ", error)

else:

    print("========================================")
    print("DATASET PREPARATION SUCCESSFUL")
    print("========================================")

    print(f"Train images : {len(list(TRAIN_IMAGES.iterdir()))}")
    print(f"Val images   : {len(list(VAL_IMAGES.iterdir()))}")

    print(f"\nDataset YAML:")
    print(yaml_file)

    print("\nReady for YOLO training.")