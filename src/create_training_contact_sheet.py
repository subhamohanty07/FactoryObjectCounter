import cv2
import os
import glob
import math
import numpy as np

INPUT_DIR = "output/training_frames"
OUTPUT_PATH = "output/training_contact_sheet.jpg"

THUMB_WIDTH = 192
THUMB_HEIGHT = 216

COLS = 6

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
)

if not files:
    raise RuntimeError("No training frames found.")

frames = []

for file in files:

    frame = cv2.imread(file)

    if frame is None:
        continue

    # Resize while keeping our contact sheet manageable
    frame = cv2.resize(
        frame,
        (THUMB_WIDTH, THUMB_HEIGHT)
    )

    # Get filename for timestamp
    name = os.path.basename(file)

    cv2.putText(
        frame,
        name,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    frames.append(frame)

rows = math.ceil(len(frames) / COLS)

sheet = np.ones(
    (
        rows * THUMB_HEIGHT,
        COLS * THUMB_WIDTH,
        3
    ),
    dtype=np.uint8
) * 255

for i, frame in enumerate(frames):

    row = i // COLS
    col = i % COLS

    y1 = row * THUMB_HEIGHT
    y2 = y1 + THUMB_HEIGHT

    x1 = col * THUMB_WIDTH
    x2 = x1 + THUMB_WIDTH

    sheet[y1:y2, x1:x2] = frame

cv2.imwrite(OUTPUT_PATH, sheet)

print(f"Frames found: {len(files)}")
print(f"Contact sheet: {OUTPUT_PATH}")