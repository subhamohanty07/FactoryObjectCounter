import cv2
import os

VIDEO_PATH = "input/sample.mp4"
OUTPUT_DIR = "output/frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps

print(f"FPS: {fps}")
print(f"Frames: {total_frames}")
print(f"Resolution: {width} x {height}")
print(f"Duration: {duration:.2f} seconds")

# Extract one frame every 5 seconds
interval_seconds = 5
frame_interval = int(fps * interval_seconds)

frame_number = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_interval == 0:
        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_number:05d}.jpg"
        )

        cv2.imwrite(filename, frame)
        saved += 1

    frame_number += 1

cap.release()

print(f"\nSaved {saved} frames to: {OUTPUT_DIR}")