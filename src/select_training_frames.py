import cv2
import os

VIDEO_PATH = "input/sample.mp4"
OUTPUT_DIR = "output/training_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select one frame every 1 second
INTERVAL_SECONDS = 1

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

frame_interval = int(fps * INTERVAL_SECONDS)

frame_number = 0
saved = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_interval == 0:

        timestamp = frame_number / fps

        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_number:05d}_{timestamp:03.0f}s.jpg"
        )

        cv2.imwrite(filename, frame)

        saved += 1

    frame_number += 1

cap.release()

print("Training frame selection complete.")
print(f"FPS: {fps}")
print(f"Duration: {duration:.2f} seconds")
print(f"Total video frames: {total_frames}")
print(f"Frames selected: {saved}")
print(f"Output folder: {OUTPUT_DIR}")