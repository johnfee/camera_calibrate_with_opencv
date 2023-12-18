import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

class UsbCamera(object):
    def __init__(self, camera_idx: int = 0) -> None:
        self.camera_idx = camera_idx
        self.image_idx = 0
        self.cap = cv2.VideoCapture(camera_idx)

        print(f"WIDTH   : {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"HEIGHT  : {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"FPS     : {self.cap.get(cv2.CAP_PROP_FPS)}")

    def capture_snapshot(self, frame):
        file_path = os.path.join("boards", f"img_{self.image_idx}.jpg")
        print(file_path)
        ret = cv2.imwrite(file_path, frame)
        if ret:
            self.image_idx += 1
            print(f"--> Capture {self.image_idx} images.")
        else:
            print(f"--> Failure Capture images.")

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()