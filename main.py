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
    elif mode == 2:
        output = frame
    elif mode == 3:
        output = cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
    elif mode == 4:
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode == 5:
        output = cv2.GaussianBlur(frame, (15, 15), 0)
    else:
        output = cv2.Canny(frame, 100, 200)

    cv2.imshow('Manipulated Webcam', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        mode = (mode + 1) % 6
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
