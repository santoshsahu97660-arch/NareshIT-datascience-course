import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -------- RED --------
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # -------- BLUE --------
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # -------- GREEN --------
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # -------- YELLOW --------
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine all masks
    combined_mask = mask_red + mask_blue + mask_green + mask_yellow
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Show all
    cv2.imshow("Original", frame)
    cv2.imshow("Multi Color Detection", result)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
