import cv2
from datetime import datetime

from src.filters import FILTERS


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    mode = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = FILTERS[mode](frame)
        cv2.imshow("Manipulated Webcam", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            mode = (mode + 1) % len(FILTERS)
        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.png"
            cv2.imwrite(filename, output)
            print(f"Saved {filename}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()