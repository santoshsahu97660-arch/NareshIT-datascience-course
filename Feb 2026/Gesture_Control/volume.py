# ------------------ HIDE TENSORFLOW WARNINGS ------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------ IMPORTS ------------------
import cv2
import mediapipe as mp
import numpy as np
import math
import screen_brightness_control as sbc

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ------------------ VOLUME SETUP ------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# ------------------ MEDIAPIPE SETUP ------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

def findDistance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# ------------------ MAIN LOOP ------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) >= 13:
                thumb = lmList[4]
                index = lmList[8]
                middle = lmList[12]

                # Draw circles
                cv2.circle(img, thumb, 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, index, 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, middle, 8, (0, 255, 0), cv2.FILLED)

                # Volume Control (Thumb + Index)
                length_vol = findDistance(thumb, index)
                vol = np.interp(length_vol, [30, 200], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                volPercent = int(np.interp(length_vol, [30, 200], [0, 100]))

                # Brightness Control (Thumb + Middle)
                length_bright = findDistance(thumb, middle)
                bright = np.interp(length_bright, [30, 200], [0, 100])
                sbc.set_brightness(int(bright))

                brightPercent = int(bright)

                # Display Text
                cv2.putText(img, f'Volume: {volPercent}%', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                cv2.putText(img, f'Brightness: {brightPercent}%', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)

    cv2.imshow("Gesture Volume & Brightness Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()