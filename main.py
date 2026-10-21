import cv2
from datetime import datetime

import numpy as np

NUM_MODES = 11

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Haar-Cascade not accessible")

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
if eye_cascade.empty():
    raise RuntimeError("Eye-Cascade not accessible")


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

def bulge(img, cx, cy, radius, strength=0.5):
    h, w = img.shape[:2]
    x0, x1 = max(cx - radius, 0), min(cx + radius, w)
    y0, y1 = max(cy - radius, 0), min(cy + radius, h)
    if x1 <= x0 or y1 <= y0:
        return img

    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    dx = (xs - cx).astype(np.float32)
    dy = (ys - cy).astype(np.float32)
    r = np.sqrt(dx**2 + dy**2) / radius

    scale = np.where(r < 1, 1 - strength * (1 - r) ** 2, 1.0).astype(np.float32)

    map_x = (cx + dx * scale).astype(np.float32)
    map_y = (cy + dy * scale).astype(np.float32)

    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return out

def enlarge_eyes(frame, strength=0.6):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(w // 8, w // 8)
        )
        for (ex, ey, ew, eh) in eyes[:2]:
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            output = bulge(output, cx, cy, int(ew * 0.9), strength)

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
    elif mode == 9:
        kernel = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]])
        output = cv2.transform(frame, kernel)
    else:
        output = enlarge_eyes(frame)

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