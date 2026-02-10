import cv2

# =============================
# PATHS (CHANGE ONLY THESE)
# =============================

# XML file path
fullbody_xml_path = r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\Full body detection\haarcascade_fullbody.xml"

# Video file path
video_path = r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\Full body detection\6387-191695740_small.mp4"

# =============================
# LOAD CASCADE
# =============================
fullbody_cascade = cv2.CascadeClassifier(fullbody_xml_path)

if fullbody_cascade.empty():
    print("❌ ERROR: Full body XML not loaded")
    exit()

# =============================
# LOAD VIDEO
# =============================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: Video not opened")
    exit()

print("✅ Full body detection started...")

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (960, 540))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =============================
    # FULL BODY DETECTION
    # =============================
    bodies = fullbody_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(60, 120)
    )

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Person",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Full Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
