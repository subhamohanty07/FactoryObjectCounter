import cv2
import os
import math

VIDEO_PATH = "input/sample.mp4"
OUTPUT_PATH = "output/timeline_contact_sheet.jpg"

# -----------------------------------
# Settings
# -----------------------------------

SAMPLE_EVERY_SECONDS = 2

THUMB_WIDTH = 240
THUMB_HEIGHT = 270

COLS = 4

# -----------------------------------
# Open video
# -----------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_interval = int(fps * SAMPLE_EVERY_SECONDS)

frames = []

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_interval == 0:

        frame_copy = frame.copy()

        # Add timestamp
        seconds = frame_number / fps

        cv2.putText(
            frame_copy,
            f"{seconds:.1f} sec",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        frame_copy = cv2.resize(
            frame_copy,
            (THUMB_WIDTH, THUMB_HEIGHT)
        )

        frames.append(frame_copy)

    frame_number += 1

cap.release()

# -----------------------------------
# Create contact sheet
# -----------------------------------

rows = math.ceil(len(frames) / COLS)

sheet = cv2.imread(
    # create blank image below instead
    VIDEO_PATH
)

sheet = 255 * __import__("numpy").ones(
    (
        rows * THUMB_HEIGHT,
        COLS * THUMB_WIDTH,
        3
    ),
    dtype="uint8"
)

for i, frame in enumerate(frames):

    row = i // COLS
    col = i % COLS

    y1 = row * THUMB_HEIGHT
    y2 = y1 + THUMB_HEIGHT

    x1 = col * THUMB_WIDTH
    x2 = x1 + THUMB_WIDTH

    sheet[y1:y2, x1:x2] = frame

os.makedirs("output", exist_ok=True)

cv2.imwrite(OUTPUT_PATH, sheet)

print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Sample interval: {SAMPLE_EVERY_SECONDS} seconds")
print(f"Frames in contact sheet: {len(frames)}")
print(f"Saved to: {OUTPUT_PATH}")