from pathlib import Path
import random
import math

import cv2
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "output" / "annotation_check"
OUTPUT_FILE = OUTPUT_DIR / "annotation_preview.jpg"


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 6
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

if not images:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

selected_images = random.sample(
    images,
    min(NUM_IMAGES, len(images))
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PLOT
# ============================================================

columns = 2
rows = math.ceil(len(selected_images) / columns)

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(14, 18)
)

# Make axes iterable even if only one row
if hasattr(axes, "flat"):
    axes = axes.flat
else:
    axes = [axes]


total_boxes = 0


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

for ax, image_path in zip(axes, selected_images):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    boxes_in_image = 0

    if label_path.exists():

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates → pixels
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Keep coordinates inside image
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            cv2.putText(
                image,
                f"object ({class_id})",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            boxes_in_image += 1

    total_boxes += boxes_in_image

    ax.imshow(image)
    ax.set_title(
        f"{image_path.name}\n"
        f"Objects: {boxes_in_image}"
    )
    ax.axis("off")


# Hide unused axes
for ax in list(axes)[len(selected_images):]:
    ax.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESULT
# ============================================================

print("========================================")
print("ANNOTATION VERIFICATION COMPLETE")
print("========================================")

print(f"Images checked : {len(selected_images)}")
print(f"Objects shown  : {total_boxes}")
print(f"\nPreview saved to:")
print(OUTPUT_FILE)