"""
GenerateArucoMaker.py

generate_board = CharucoBoard(dictionary type)

dictionary type
    dictionary = {
        "4X4_50": aruco.DICT_4X4_50,
        "5X5_50": aruco.DICT_5X5_50,
        "6X6_50": aruco.DICT_6X6_50,
        "7X7_50": aruco.DICT_7X7_50,
    }
"""
import os
import sys

import cv2
from CharucoBoard import CharucoBoard
from util import load_images


def main():
    # Create Charucoboard object
    board = CharucoBoard(
        dict_type="4X4_50",
        square_row=5,
        square_col=7,
        square_length=0.04,  # meter
        marker_length=0.02,  # meter
    )

    board.show_charucoboard_info()

    img = load_images()
    board.camera_calibrate(img)

    # Exit on any key
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
