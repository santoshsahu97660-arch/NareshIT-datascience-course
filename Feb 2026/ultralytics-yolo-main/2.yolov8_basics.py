from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

# predict on an image
detection_output = model.predict(source=r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\ultralytics-yolo-main\1 jpeg image.jpeg", conf=0.25, save=True) 

# Display tensor array
print(detection_output)

# Display numpy array
# print(detection_output[0].numpy())
