import os
import sys
import traceback
import cv2
from cv2 import aruco
import numpy as np
import yaml
import glob
import matplotlib.pyplot as plt

from util import load_images


class CharucoBoard(object):
    def __init__(self, dict_type: str, square_row: int, square_col: int, square_length: float, marker_length: float):
        self.dictionary = {
            "4X4_50": aruco.DICT_4X4_50,
            "5X5_50": aruco.DICT_5X5_50,
            "6X6_50": aruco.DICT_6X6_50,
            "7X7_50": aruco.DICT_7X7_50,
        }

        # aruco marker
        self.dict_type = dict_type
        self.dict_aruco = aruco.getPredefinedDictionary(self.dictionary[dict_type])
        self.parameter = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dict_aruco, self.parameter)

        # charucoboard
        self.pattern_size = (square_row, square_col)
        self.square_length = square_length
        self.marker_length = marker_length
        self.board = aruco.CharucoBoard(
            self.pattern_size, self.square_length, self.marker_length, self.dict_aruco)
        self.board_detector = aruco.CharucoDetector(self.board)

        # charucoboard property
        self.board_size = self.board.getChessboardSize()
        self.board_square_length = self.board.getSquareLength()
        self.board_marker_length = self.board.getMarkerLength()
        self.board_objp = np.array(self.board.getChessboardCorners())

    def show_charucoboard_info(self) -> None:
        print("--- Charucoboard property info ---")
        print(f"Board Size : Row {self.board_size[0]} / Column {self.board_size[1]}")
        print(f"Square size [mm] : {round(self.board_square_length, 3)}")
        print(f"Marker size [mm] : {round(self.board_marker_length, 3)}")
        print("----------------------------------\n")

    def generate_charucoboard(self):
        file_path = f"boards\\board_{self.dict_type}.png"
        dpi = 80
        mm_to_inch = 1 / 25.4
        square_length_mm = 1000 * self.square_length

        img_row_pix = round(self.pattern_size[0] * mm_to_inch * square_length_mm * dpi)
        img_col_pix = round(self.pattern_size[1] * mm_to_inch * square_length_mm * dpi)
        board_img = self.board.generateImage((img_row_pix, img_col_pix))
        result = cv2.imwrite(file_path, board_img)
        if result:
            print("Success generating charucoboard.")
        else:
            print("Failure generating charucoboard.")

        return result, board_img

    def draw_detect_board(self, frame):
        # convert gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # charucoboard detecting and drawing
        charuco_corners, charuco_ids, _, _ = self.board_detector.detectBoard(gray)
        frame_ = aruco.drawDetectedCornersCharuco(gray, charuco_corners, charuco_ids)
        return frame_

    def camera_calibrate(self, images_list):
        """
        Charuco base pose estimation. and camera calibration
        """
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        objp = np.zeros(((self.pattern_size[0] - 1) * (self.pattern_size[1] - 1), 3), np.float32)
        objp[:, :2] = np.mgrid[0:(self.pattern_size[1] - 1), 0:(self.pattern_size[0] - 1)].T.reshape(-1, 2)

        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for idx in range(len(images_list)):
            print(f"--> Processing image {idx}")

            frame = cv2.imread(f"boards\\img_{idx}.jpg")
            cv2.imshow(f"load image {idx}", frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detection charucoboard
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.board_detector.detectBoard(frame)

            if len(marker_corners) > 0:
                # SUB PIXEL DETECTION
                objpoints.append(self.board_objp)

                # refining pixel coordinates for given 2D points
                corners2 = cv2.cornerSubPix(gray, charuco_corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame, self.pattern_size, corners2, True)
                cv2.imshow(f"detect corner image {idx}", frame)
                cv2.waitKey(1000)
                imgpoints.append(corners2)

            imsize = gray.shape[::-1]
            # print(objpoints[idx])
            # print(imgpoints[idx])

        print("--> CAMERA CALIBRATE WITH CharucoBoard START")
        ret, cam_mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)

        print(ret)
        print("--> Camera matrix : \n")
        print(cam_mtx)
        print("--> dist coeffs : \n")
        print(dist_coeff)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)
        camera_calibration_dict = {
            'ret': ret,
            'mtx': cam_mtx,
            'dist': dist_coeff,
            'rvecs': rvecs,
            'tvecs': tvecs
        }

        dir_config = os.path.join(os.getcwd(), 'config\\camera_calibration')
        with open(dir_config, 'w') as file:
            documents = yaml.dump(camera_calibration_dict, file)

        print("--> CAMERA CALIBRATE WITH CharucoBoard END")