import cv2


def main():
    input_video = r"C:\Curriculum\factory_object_counter\input\sample.mp4"

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    # Read the first frame
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read frame.")
        cap.release()
        return

    height, width = frame.shape[:2]

    print(f"Video resolution: {width} x {height}")

    # ROI coordinates
    x1 = 172
    y1 = 604
    x2 = 550
    y2 = 1079

    # Draw ROI
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    # Add label
    cv2.putText(
        frame,
        "COUNTING ROI",
        (x1, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Show coordinates
    print("\nROI coordinates:")
    print(f"Top-left     : ({x1}, {y1})")
    print(f"Bottom-right : ({x2}, {y2})")

    # Save preview
    output_path = (
        r"C:\Curriculum\factory_object_counter"
        r"\output\roi_preview.jpg"
    )

    cv2.imwrite(output_path, frame)

    print(f"\nROI preview saved to:")
    print(output_path)

    cap.release()


if __name__ == "__main__":
    main()