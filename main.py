import cv2

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
    else:
        output = frame

    cv2.imshow('Manipulated Webcam', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        mode = (mode + 1) % 3
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
