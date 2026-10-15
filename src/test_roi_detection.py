import cv2
from ultralytics import YOLO


def main():
    print("=" * 60)
    print("FACTORY OBJECT COUNTER - ROI DETECTION TEST")
    print("=" * 60)

    # Trained model
    model = YOLO(
        r"C:\Curriculum\factory_object_counter\runs\detect\output\training\factory_object_model\weights\best.pt"
    )

    # Input video
    input_video = (
        r"C:\Curriculum\factory_object_counter\input\sample.mp4"
    )

    # Output video
    output_video = (
        r"C:\Curriculum\factory_object_counter"
        r"\output\roi_detection.avi"
    )

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Our confirmed ROI
    x1 = 172
    y1 = 604
    x2 = 550
    y2 = 1079

    roi_width = x2 - x1
    roi_height = y2 - y1

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Crop only the ROI
        roi = frame[y1:y2, x1:x2]

        # Run YOLO only on ROI
        results = model.predict(
            roi,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # Draw detections on ROI
        annotated_roi = result.plot()

        # Put annotated ROI back into original frame
        frame[y1:y2, x1:x2] = annotated_roi

        # Draw ROI boundary
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "COUNTING ROI",
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        # Number of detections
        detection_count = 0

        if result.boxes is not None:
            detection_count = len(result.boxes)

        cv2.putText(
            frame,
            f"Detections in ROI: {detection_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        writer.write(frame)

        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

    cap.release()
    writer.release()

    print("\n" + "=" * 60)
    print("ROI DETECTION TEST COMPLETED")
    print("=" * 60)
    print("Output:")
    print(output_video)


if __name__ == "__main__":
    main()