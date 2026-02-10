import numpy as np
import cv2

# Load Haar Cascade
face_classifier = cv2.CascadeClassifier(
    r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\haarcascade_frontalface_default.xml"
)

# Load image
image = cv2.imread(
    r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\10th feb 2026\opencv\momita.jpg"
)

if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No faces found!")
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 225), 2)

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
