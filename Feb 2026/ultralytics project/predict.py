from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run prediction
results = model("bus (1).jpg", save=True)

print("Prediction Done ✅")