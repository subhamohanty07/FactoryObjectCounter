from ultralytics import YOLO
import cv2
import os

VIDEO_PATH = "input/sample.mp4"
OUTPUT_DIR = "output/pretrained_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load a small pretrained YOLO model
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Test frames at these times
test_times = [5, 20, 35, 50, 65, 80, 95, 105]

print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")
print()
print("Running pretrained YOLO test...")
print()

for seconds in test_times:

    frame_number = int(seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame at {seconds}s")
        continue

    # Run detection
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Draw detections
    annotated = result.plot()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"frame_{seconds:03d}s.jpg"
    )

    cv2.imwrite(output_path, annotated)

    print(
        f"{seconds:3d}s -> "
        f"{len(result.boxes)} detections -> "
        f"{output_path}"
    )

cap.release()

print()
print("Baseline detection test complete.")