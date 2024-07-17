import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set parameters for the Charuco board
squaresX = 15  # Number of squares in X direction
squaresY = 7  # Number of squares in Y direction
squareLength = 0.04  # Square side length (in meters)
markerLength = 0.02  # Marker side length (in meters)

# Create Charuco board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, aruco_dict)

# Generate Charuco board image
board_img = charuco_board.generateImage((200 * squaresX, 200 * squaresY))

# Threshold the image to make it pure black and white
_, binary_img = cv2.threshold(board_img, 127, 255, cv2.THRESH_BINARY)

# Display the image
cv2.imshow("Generated Charuco Board", binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save as SVG
fig, ax = plt.subplots(figsize=(squaresX, squaresY))
ax.imshow(binary_img, cmap='gray')
ax.axis('off')

# Save figure as SVG
plt.savefig("charuco_board.svg", format='svg')
plt.savefig("charuco_board.png", format='png')
