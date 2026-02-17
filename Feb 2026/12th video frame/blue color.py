import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue color range
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])

    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Blue Color", blue)

    key = cv2.waitKey(1)
    if key == 27:   # ESC key
        break

cap.release()
cv2.destroyAllWindows()
