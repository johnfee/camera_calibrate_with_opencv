import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt


class ArucoMarker(object):
    def __init__(self, dict_type: str) -> None:
        self.dictionary = {
            "4X4_50": aruco.DICT_4X4_50,
            "5X5_50": aruco.DICT_5X5_50,
            "6X6_50": aruco.DICT_6X6_50,
            "7X7_50": aruco.DICT_7X7_50,
        }

        self.dict_type = dict_type
        self.SAVE_MARK_NAME = f"marker_{dict_type}.png"
        self.dict_aruco = aruco.getPredefinedDictionary(self.dictionary[dict_type])
        self.parameter = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dict_aruco, self.parameter)

    def generate_marker(self, marker_id: int = 0, marker_size_pix: int or float = 50):
        marker_img = aruco.generateImageMarker(self.dict_aruco, marker_id, marker_size_pix)
        result = cv2.imwrite(f"marker\\id_{marker_id}_" + self.SAVE_MARK_NAME, marker_img)
        if result:
            print("Success generating Aruco marker.")
        else:
            print("Failure generating Aruco marker.")
        return result, marker_img

    def draw_detected_markers(self, frame):
        # convert gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # marker detecting and drawing
        corners, ids, _ = self.detector.detectMarkers(gray)
        frame_ = aruco.drawDetectedMarkers(frame, corners, ids)
        return frame_
