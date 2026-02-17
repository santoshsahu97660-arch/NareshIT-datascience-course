import cv2

cap = cv2.VideoCapture("C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\13th- Img frame, object_tracking from video\highway.mp4")

if not cap.isOpened():
    print("Video not found!")
    exit()

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = object_detector.apply(frame)

    # remove noise
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
