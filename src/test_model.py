from ultralytics import YOLO


def main():
    print("=" * 50)
    print("FACTORY OBJECT COUNTER - MODEL TEST")
    print("=" * 50)

    # Trained model
    model = YOLO(
        r"C:\Curriculum\factory_object_counter\runs\detect\output\training\factory_object_model\weights\best.pt"
    )

    # Test video
    video_path = r"C:\Curriculum\factory_object_counter\input\sample.mp4"

    print("\nRunning detection...")
    print("Video:", video_path)

    model.predict(
        source=video_path,
        conf=0.25,
        device="cpu",
        save=True,
        project=r"C:\Curriculum\factory_object_counter\output",
        name="custom_model_detection",
        exist_ok=True
    )

    print("\n" + "=" * 50)
    print("DETECTION COMPLETED")
    print("=" * 50)

    print(
        r"Output folder: C:\Curriculum\factory_object_counter\output\custom_model_detection"
    )


if __name__ == "__main__":
    main()