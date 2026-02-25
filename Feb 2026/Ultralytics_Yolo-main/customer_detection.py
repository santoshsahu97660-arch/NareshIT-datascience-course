import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')   # <-- FIX

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # inference
    results = model(frame, conf=0.4)

    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Customer Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()