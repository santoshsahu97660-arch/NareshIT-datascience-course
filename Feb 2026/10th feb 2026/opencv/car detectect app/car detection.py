import cv2
import time
import math

# =============================
# LOAD HAAR CASCADE
# =============================
car_classifier_path = r'C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\car detectect app\haarcascade_car.xml'
car_classifier = cv2.CascadeClassifier(car_classifier_path)

if car_classifier.empty():
    print("❌ Cascade not loaded")
    exit()

video_path = r'C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\car detectect app\istockphoto-981340772-640_adpp_is.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video not opened")
    exit()

# =============================
# SPEED SETTINGS
# =============================
PIXEL_TO_METER = 0.04   # smaller = slower & stable speed
prev_positions = {}
prev_speeds = {}

# =============================
# FULLSCREEN WINDOW
# =============================
cv2.namedWindow("Car Speed & Traffic", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Car Speed & Traffic",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("Running...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    roi_y = int(h * 0.35)
    roi = gray[roi_y:h, 0:w]

    cars = car_classifier.detectMultiScale(
        roi,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(20, 20)
    )

    now = time.time()
    cars_in_frame = len(cars)

    # =============================
    # TRAFFIC STATUS
    # =============================
    if cars_in_frame < 8:
        traffic = "LOW"
        tcolor = (0, 255, 0)
    elif cars_in_frame < 18:
        traffic = "MEDIUM"
        tcolor = (0, 255, 255)
    else:
        traffic = "HIGH"
        tcolor = (0, 0, 255)

    # =============================
    # INFO PANEL
    # =============================
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, "Car Speed & Traffic System", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Traffic: {traffic}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcolor, 2)

    # =============================
    # DETECTION + SPEED
    # =============================
    for (x, y, w1, h1) in cars:
        cx = x + w1 // 2
        cy = y + h1 // 2 + roi_y
        obj_id = f"{cx//40}_{cy//40}"

        speed = 0.0
        if obj_id in prev_positions:
            px, py, pt = prev_positions[obj_id]
            dp = math.hypot(cx - px, cy - py)
            dm = dp * PIXEL_TO_METER
            dt = now - pt
            if dt > 0:
                speed = (dm / dt) * 3.6
                if obj_id in prev_speeds:
                    speed = (prev_speeds[obj_id] + speed) / 2  # smoothing

        prev_positions[obj_id] = (cx, cy, now)
        prev_speeds[obj_id] = speed

        # =============================
        # SPEED COLOR LOGIC
        # =============================
        if speed > 70:
            color = (0, 0, 255)      # RED
        elif speed > 40:
            color = (0, 255, 255)    # YELLOW
        else:
            color = (0, 255, 0)      # GREEN

        label = f"{int(speed)} km/h"

        # BOX
        cv2.rectangle(frame,
                      (x, y + roi_y),
                      (x + w1, y + h1 + roi_y),
                      color, 2)

        # LABEL BACKGROUND
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame,
                      (x, y + roi_y - th - 10),
                      (x + tw + 6, y + roi_y),
                      color, -1)

        cv2.putText(frame, label,
                    (x + 3, y + roi_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)

    cv2.imshow("Car Speed & Traffic", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
