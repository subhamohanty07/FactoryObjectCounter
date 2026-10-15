import cv2
from ultralytics import YOLO


def main():
    print("=" * 60)
    print("FACTORY OBJECT COUNTER - OBJECT TRACKING")
    print("=" * 60)

    # Trained YOLO model
    model = YOLO(
        r"C:\Curriculum\factory_object_counter\runs\detect\output\training\factory_object_model\weights\best.pt"
    )

    # Input video
    input_video = r"C:\Curriculum\factory_object_counter\input\sample.mp4"

    # Output video
    output_video = (
        r"C:\Curriculum\factory_object_counter\output\tracked_objects.avi"
    )

    # Open video
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("ERROR: Could not open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Resolution: {width} x {height}")
    print(f"Total frames: {total_frames}")

    # Video writer
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

        # Track objects
        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # Draw tracking results
        annotated_frame = result.plot()

        writer.write(annotated_frame)

        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

    cap.release()
    writer.release()

    print("\n" + "=" * 60)
    print("TRACKING COMPLETED")
    print("=" * 60)
    print(f"Output video:")
    print(output_video)


if __name__ == "__main__":
    main()