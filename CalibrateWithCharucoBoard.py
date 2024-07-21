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
        square_row=7,
        square_col=5,
        square_length_mm=40,  # mm meter
        marker_length_mm=30,  # mm meter
    )

    board.show_charucoboard_info()

    img = load_images()
    board.camera_calibrate(img)


if __name__ == "__main__":
    main()
