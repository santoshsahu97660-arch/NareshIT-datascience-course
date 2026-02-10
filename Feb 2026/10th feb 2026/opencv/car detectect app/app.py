from ultralytics import YOLO
import cv2
import time
import math

# =============================
# CONFIG
# =============================
video_path = r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\car detectect app\istockphoto-981340772-640_adpp_is.mp4"
output_path = "final_output.mp4"

PIXEL_TO_METER = 0.05     # camera dependent (approx)
CONF_THRESHOLD = 0.3

# COCO vehicle classes
labels = {
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}

# =============================
# LOAD MODEL
# =============================
model = YOLO("yolov8n.pt")   # change to yolov8s.pt for more accuracy

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Video not opened")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =============================
# VARIABLES
# =============================
prev_positions = {}
vehicle_count = 0
counted_ids = set()
count_line_y = int(height * 0.65)

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    vehicles_in_frame = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in labels:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            vehicles_in_frame += 1
            label = labels[cls]

            # -------- UNIQUE ID (simple tracking) --------
            obj_id = f"{cls}_{cx//20}_{cy//20}"

            # -------- SPEED CALCULATION --------
            speed_text = ""
            if obj_id in prev_positions:
                px, py, pt = prev_positions[obj_id]
                dist_pixels = math.hypot(cx - px, cy - py)
                dist_meters = dist_pixels * PIXEL_TO_METER
                time_diff = current_time - pt

                if time_diff > 0:
                    speed_kmph = (dist_meters / time_diff) * 3.6
                    speed_text = f"{int(speed_kmph)} km/h"

            prev_positions[obj_id] = (cx, cy, current_time)

            # -------- COUNTING --------
            if abs(cy - count_line_y) < 6:
                if obj_id not in counted_ids:
                    vehicle_count += 1
                    counted_ids.add(obj_id)

            # -------- DRAWING --------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            if speed_text:
                cv2.putText(
                    frame,
                    speed_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # -------- TRAFFIC DENSITY --------
    if vehicles_in_frame < 10:
        traffic = "LOW"
        tcolor = (0, 255, 0)
    elif vehicles_in_frame < 25:
        traffic = "MEDIUM"
        tcolor = (0, 255, 255)
    else:
        traffic = "HIGH"
        tcolor = (0, 0, 255)

    # -------- UI --------
    cv2.line(frame, (0, count_line_y), (width, count_line_y), (255, 0, 0), 2)

    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Traffic: {traffic}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, tcolor, 2)

    out.write(frame)
    cv2.imshow("YOLO Vehicle Detection - FINAL", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Output saved as:", output_path)
