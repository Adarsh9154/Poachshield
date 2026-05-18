from ultralytics import YOLO

# Load weapon model
model = YOLO("weapon_detection_best.pt")

# Test image
image_path = "test02.webp"

# Run detection
results = model(
    image_path,
    conf=0.10,
    imgsz=1280,
    show=True
)

# Print detections
boxes = results[0].boxes

print("\nDetections:\n")

for box in boxes:

    cls_id = int(box.cls[0])

    label = model.names[cls_id]

    confidence = float(box.conf[0])

    print(f"{label} -> {confidence:.2f}")