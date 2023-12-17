"""
GenerateArucoMaker.py

generate_maker = ArucoMarker(dictionary type)

dictionary type
    dictionary = {
        "4X4_50": aruco.DICT_4X4_50,
        "5X5_50": aruco.DICT_5X5_50,
        "6X6_50": aruco.DICT_6X6_50,
        "7X7_50": aruco.DICT_7X7_50,
    }
"""

import cv2
from ArucoMarker import ArucoMarker

def main():
    # Create Aruco Marker object
    marker = ArucoMarker("4X4_50")
    # Generate Aruco Marker (marker id, size of marker)
    _, gen_marker_img = marker.generate_marker(marker_id=2, marker_size_pix=100)
    # Display the image to us
    cv2.imshow("Generate Aruco Marker", gen_marker_img)
    # Exit on any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
