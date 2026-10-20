import cv2
from datetime import datetime

import numpy as np

NUM_MODES = 10

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Haar-Cascade konnte nicht geladen werden")


def pixelate(img, blocks_w=32, blocks_h=24):
    h, w = img.shape[:2]
    small = cv2.resize(img, (blocks_w, blocks_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelate_faces(frame, block_size=12):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        blocks_w = max(1, w // block_size)
        blocks_h = max(1, h // block_size)
        output[y:y + h, x:x + w] = pixelate(roi, blocks_w, blocks_h)

    return output


cap = cv2.VideoCapture(0)

mode = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if mode == 0:
        output = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    elif mode == 1:
        output = cv2.bitwise_not(frame)
    elif mode == 2:
        output = frame
    elif mode == 3:
        output = cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
    elif mode == 4:
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode == 5:
        output = cv2.GaussianBlur(frame, (15, 15), 0)
    elif mode == 6:
        output = cv2.Canny(frame, 100, 200)
    elif mode == 7:
        output = pixelate(frame)
    elif mode == 8:
        output = pixelate_faces(frame)
    else:
        kernel = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]])
        output = cv2.transform(frame, kernel)

    cv2.imshow('Manipulated Webcam', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        mode = (mode + 1) % NUM_MODES
    elif key == ord('s'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.png'
        cv2.imwrite(filename, output)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()