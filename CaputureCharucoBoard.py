import os
import cv2

from Camera import UsbCamera

def main():
    cam = UsbCamera()

    while True:
        ret, frame = cam.cap.read()
        cv2.imshow("raw", frame)

        key = cv2.waitKey(2)
        if key == 27:
            break
        elif key == 13 or key == 32:
            cam.capture_snapshot(frame)

    cam.close_camera()


if __name__ == "__main__":
    main()