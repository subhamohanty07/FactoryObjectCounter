import cv2
from ultralytics import YOLO
from pathlib import Path


def main():
    print("=" * 60)
    print("FACTORY OBJECT COUNTER - ROI LINE CROSSING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. PROJECT PATHS
    # --------------------------------------------------
    project_dir = Path(r"C:\Curriculum\factory_object_counter")

    model_path = (
        project_dir
        / "runs"
        / "detect"
        / "output"
        / "training"
        / "factory_object_model"
        / "weights"
        / "best.pt"
    )

    input_video = project_dir / "input" / "sample.mp4"

    tracker_config = (
        project_dir / "src" / "custom_bytetrack.yaml"
    )

    output_video = (
        project_dir / "output" / "object_counting.avi"
    )

    # --------------------------------------------------
    # 2. COUNTING LINE CONFIGURATION
    # --------------------------------------------------
    # IMPORTANT:
    # Change these points to match the desired crossing
    # location in your factory video.
    #
    # Video resolution: 960 x 1080
    # Coordinates are (x, y).

    LINE_START = (320, 720)
    LINE_END = (235, 1075)

    # Count only one crossing direction.
    #
    # "negative_to_positive":
    #     Count when the point moves from the negative
    #     side of the directed line to its positive side.
    #
    # "positive_to_negative":
    #     Count in the opposite direction.
    COUNT_DIRECTION = "negative_to_positive"

    # Ignore tiny movements near the line.
    CROSSING_MARGIN = 5.0

    # --------------------------------------------------
    # 3. YOLO SETTINGS
    # --------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.10
    NMS_IOU_THRESHOLD = 0.45

    # --------------------------------------------------
    # 4. VALIDATE FILES
    # --------------------------------------------------
    for path in (model_path, input_video, tracker_config):
        if not path.is_file():
            print(f"ERROR: File not found: {path}")
            return

    if COUNT_DIRECTION not in (
        "negative_to_positive",
        "positive_to_negative",
    ):
        print("ERROR: Invalid COUNT_DIRECTION.")
        return

    output_video.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 5. LOAD MODEL
    # --------------------------------------------------
    print("\nLoading custom YOLO model...")
    model = YOLO(str(model_path))
    print("Model loaded successfully.")

    # --------------------------------------------------
    # 6. OPEN VIDEO
    # --------------------------------------------------
    cap = cv2.VideoCapture(str(input_video))

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
    print(f"Counting line: {LINE_START} -> {LINE_END}")
    print(f"Direction: {COUNT_DIRECTION}")

    # Validate the configured line.
    for x, y in (LINE_START, LINE_END):
        if not (0 <= x < width and 0 <= y < height):
            print("ERROR: Counting-line coordinates are outside the frame.")
            cap.release()
            return

    if LINE_START == LINE_END:
        print("ERROR: LINE_START and LINE_END must be different.")
        cap.release()
        return

    # --------------------------------------------------
    # 7. CREATE OUTPUT VIDEO
    # --------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        return

    # --------------------------------------------------
    # 8. TRACKING AND COUNTING STATE
    # --------------------------------------------------
    # Previous signed distance from the line for each ID.
    previous_sides = {}

    # Previous bottom-center position for each ID.
    previous_positions = {}

    # IDs that have already been counted.
    counted_ids = set()

    total_count = 0
    frame_number = 0

    line_x1, line_y1 = LINE_START
    line_x2, line_y2 = LINE_END

    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1
    line_length_squared = line_dx**2 + line_dy**2

    # --------------------------------------------------
    # 9. PROCESS VIDEO
    # --------------------------------------------------
    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # YOLO detection + ByteTrack.
            # Keep the current model and tracker configuration.
            results = model.track(
                source=frame,
                persist=True,
                tracker=str(tracker_config),
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                end2end=False,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            # Draw detections and tracking IDs.
            annotated_frame = result.plot()

            # Draw the virtual counting line.
            cv2.line(
                annotated_frame,
                LINE_START,
                LINE_END,
                (0, 255, 255),
                3,
            )

            cv2.circle(
                annotated_frame,
                LINE_START,
                6,
                (255, 0, 255),
                -1,
            )

            cv2.circle(
                annotated_frame,
                LINE_END,
                6,
                (255, 0, 255),
                -1,
            )

            # --------------------------------------------------
            # 10. GET TRACKED OBJECTS
            # --------------------------------------------------
            boxes = result.boxes

            if (
                boxes is not None
                and len(boxes) > 0
                and boxes.id is not None
            ):
                xyxy_list = boxes.xyxy.cpu().tolist()
                ids = boxes.id.int().cpu().tolist()

                for box_coords, track_id in zip(
                    xyxy_list, ids
                ):
                    x1, y1, x2, y2 = box_coords

                    # Bottom-center point of the bounding box.
                    center_x = (x1 + x2) / 2.0
                    bottom_y = y2

                    # Signed side of the directed line.
                    current_side = (
                        line_dx * (bottom_y - line_y1)
                        - line_dy * (center_x - line_x1)
                    )

                    current_position = (center_x, bottom_y)

                    # Show the point used for counting.
                    cv2.circle(
                        annotated_frame,
                        (
                            int(center_x),
                            int(bottom_y),
                        ),
                        5,
                        (255, 255, 0),
                        -1,
                    )

                    # --------------------------------------------------
                    # 11. CHECK WHETHER THIS ID CROSSED THE LINE
                    # --------------------------------------------------
                    if track_id in previous_sides:
                        previous_side = previous_sides[track_id]
                        crossed = False

                        if COUNT_DIRECTION == "negative_to_positive":
                            crossed = (
                                previous_side < -CROSSING_MARGIN
                                and current_side >= CROSSING_MARGIN
                            )
                        else:
                            crossed = (
                                previous_side > CROSSING_MARGIN
                                and current_side <= -CROSSING_MARGIN
                            )

                        # Check that the crossing happened on
                        # the finite line segment, not beyond its ends.
                        if crossed:
                            denominator = (
                                previous_side - current_side
                            )

                            if abs(denominator) > 1e-9:
                                fraction = (
                                    previous_side / denominator
                                )

                                previous_x, previous_y = (
                                    previous_positions[track_id]
                                )

                                crossing_x = (
                                    previous_x
                                    + fraction
                                    * (center_x - previous_x)
                                )

                                crossing_y = (
                                    previous_y
                                    + fraction
                                    * (bottom_y - previous_y)
                                )

                                projection = (
                                    (crossing_x - line_x1) * line_dx
                                    + (crossing_y - line_y1) * line_dy
                                ) / line_length_squared

                                on_line_segment = (
                                    0.0 <= projection <= 1.0
                                )

                                if (
                                    on_line_segment
                                    and track_id not in counted_ids
                                ):
                                    counted_ids.add(track_id)
                                    total_count += 1

                                    print(
                                        f"COUNTED | ID={track_id} | "
                                        f"Total={total_count} | "
                                        f"Frame={frame_number}"
                                    )

                    # Save this ID's latest position and side.
                    previous_sides[track_id] = current_side
                    previous_positions[track_id] = current_position

            # --------------------------------------------------
            # 12. DISPLAY LIVE COUNT
            # --------------------------------------------------
            cv2.rectangle(
                annotated_frame,
                (15, 15),
                (390, 95),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated_frame,
                f"Objects Counted: {total_count}",
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                "Direction: One-way",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            writer.write(annotated_frame)

            frame_number += 1

            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} "
                    f"frames | Count={total_count}"
                )

    finally:
        cap.release()
        writer.release()

    # --------------------------------------------------
    # 13. FINAL REPORT
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("OBJECT COUNTING COMPLETED")
    print("=" * 60)
    print(f"Frames processed: {frame_number}")
    print(f"Unique IDs counted: {len(counted_ids)}")
    print(f"Final object count: {total_count}")
    print(f"Output video: {output_video}")


if __name__ == "__main__":
    main()