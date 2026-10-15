from ultralytics import YOLO


def main():
    print("=" * 50)
    print("FACTORY OBJECT COUNTER - MODEL TRAINING")
    print("=" * 50)

    # Load a small pretrained YOLO model
    model = YOLO("yolo26n.pt")

    # Train on CPU
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project="output/training",
        name="factory_object_model",
        exist_ok=True
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print("Best model:")
    print("output/training/factory_object_model/weights/best.pt")


if __name__ == "__main__":
    main()