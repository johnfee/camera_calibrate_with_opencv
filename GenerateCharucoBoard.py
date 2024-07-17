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


def main():
    # Create Charucoboard object
    board = CharucoBoard(
        dict_type="4X4_50",
        square_row=14,
        square_col=7,
        square_length_mm=1000,  # meter
        marker_length_mm=900,  # meter
    )

    # Generate Aruco Marker (marker id, size of marker)
    _, gen_board_img = board.generate_charucoboard()
    board.show_charucoboard_info()

    # Display the image to us
    cv2.imshow("Generate Charucoboard", gen_board_img)
    # Exit on any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
