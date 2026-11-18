import cv2
import numpy as np
import os
import math

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Face cascade not accessible")

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
if eye_cascade.empty():
    raise RuntimeError("Eye cascade not accessible")

SEPIA_KERNEL = np.array([[0.131, 0.534, 0.272],
                         [0.168, 0.686, 0.349],
                         [0.189, 0.769, 0.393]])


def detect_faces(gray):
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )


def pixelate(img, blocks_w=32, blocks_h=24):
    h, w = img.shape[:2]
    small = cv2.resize(img, (blocks_w, blocks_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelate_faces(frame, block_size=12):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in detect_faces(gray):
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
    r = np.sqrt(dx ** 2 + dy ** 2) / radius

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

    for (x, y, w, h) in detect_faces(gray):
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(w // 8, w // 8)
        )
        for (ex, ey, ew, eh) in eyes[:2]:
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            output = bulge(output, cx, cy, int(ew * 0.9), strength)

    return output

SUNGLASSES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "sunglasses.png"
)
SUNGLASSES = cv2.imread(SUNGLASSES_PATH, cv2.IMREAD_UNCHANGED)
if SUNGLASSES is None or SUNGLASSES.ndim != 3 or SUNGLASSES.shape[2] != 4:
    raise RuntimeError("Sunglasses image missing or has no alpha channel: " + SUNGLASSES_PATH)


def rotate_rgba(img, angle):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        img, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def overlay_rgba(frame, overlay, cx, cy):
    frame_h, frame_w = frame.shape[:2]
    over_h, over_w = overlay.shape[:2]

    x0 = cx - over_w // 2
    y0 = cy - over_h // 2

    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + over_w, frame_w), min(y0 + over_h, frame_h)
    if fx1 <= fx0 or fy1 <= fy0:
        return

    ox0, oy0 = fx0 - x0, fy0 - y0
    ox1, oy1 = ox0 + (fx1 - fx0), oy0 + (fy1 - fy0)

    patch = overlay[oy0:oy1, ox0:ox1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)

    blended = alpha * patch[:, :, :3].astype(np.float32) + (1.0 - alpha) * region
    frame[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)


def place_sunglasses(frame, cx, cy, width, angle):
    ratio = SUNGLASSES.shape[0] / SUNGLASSES.shape[1]
    width = max(1, int(width))
    height = max(1, int(width * ratio))
    resized = cv2.resize(SUNGLASSES, (width, height), interpolation=cv2.INTER_AREA)
    rotated = rotate_rgba(resized, angle)
    overlay_rgba(frame, rotated, cx, cy)


def sunglasses(frame):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in detect_faces(gray):
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(w // 8, w // 8)
        )

        if len(eyes) >= 2:
            largest = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            left, right = sorted(largest, key=lambda e: e[0])

            lx = x + left[0] + left[2] // 2
            ly = y + left[1] + left[3] // 2
            rx = x + right[0] + right[2] // 2
            ry = y + right[1] + right[3] // 2

            distance = math.hypot(rx - lx, ry - ly)
            angle = -math.degrees(math.atan2(ry - ly, rx - lx))
            place_sunglasses(
                output, (lx + rx) // 2, (ly + ry) // 2, distance * 2.3, angle
            )
        else:
            place_sunglasses(output, x + w // 2, y + int(h * 0.38), w * 0.95, 0)

    return output


def jet(frame):
    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)


def invert(frame):
    return cv2.bitwise_not(frame)


def original(frame):
    return frame


def hsv_map(frame):
    return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)


def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def rgb_split(frame, shift=8):
    b, g, r = cv2.split(frame)
    b = np.roll(b, -shift, axis=1)
    r = np.roll(r, shift, axis=1)
    return cv2.merge([b, g, r])



def blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)


def edges(frame):
    return cv2.Canny(frame, 100, 200)


def sepia(frame):
    return cv2.transform(frame, SEPIA_KERNEL)

def cartoon(frame):
    # Smooth colors while preserving edges
    color = cv2.bilateralFilter(frame, 9, 250, 250)

    # Detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edge_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        2
    )

    # Combine smooth colors with black outlines
    return cv2.bitwise_and(color, color, mask=edge_mask)


FILTERS = [
    jet,
    invert,
    original,
    hsv_map,
    grayscale,
    blur,
    edges,
    pixelate,
    pixelate_faces,
    sepia,
    enlarge_eyes,
    rgb_split,
    sunglasses,
    cartoon
]