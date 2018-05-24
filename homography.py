
from collections import Iterable
import numpy as np
import cv2

_selectedPoints = []
_currentImage = None
_getPointsWindowName = "Select Target Positions"

# Callback for getting the mouse click locations
def getClick(event, x, y, _, __):
    global _selectedPoints
    global _currentImage

    if event == cv2.EVENT_LBUTTONDOWN:
            _selectedPoints.append((float(x), float(y)))
            cv2.circle(_currentImage, (x, y), 5, (0, 255, 0))
            cv2.imshow(_getPointsWindowName, _currentImage)


# Gets mouse click locations on the specified image (press 'q') to finish
def getTargetLocations(image):
    global _selectedPoints
    global _currentImage

    _selectedPoints = []
    _currentImage = image

    cv2.namedWindow(_getPointsWindowName)
    cv2.setMouseCallback(_getPointsWindowName, getClick)

    while True:
            cv2.imshow(_getPointsWindowName, image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    return _selectedPoints


def getPhysicalCoordinates(sourcePoints):
	INVALID_COORDS_MESSAGE = "Invalid Coordinates. Input as x, y"
	destinationPoints = []

	for p in sourcePoints:
		coords = []
		while not isinstance(coords, Iterable) or len(coords) != 2:
			try:
				coords = input(str(p) + ": ")
				if len(coords) != 2:
					print INVALID_COORDS_MESSAGE
				else:
					destinationPoints.append(coords)
			except (NameError, TypeError):
				print INVALID_COORDS_MESSAGE

	return destinationPoints


def getHomographyMatrix(sourcePoints, destinationPoints):
	H, _ = cv2.findHomography(np.asarray(sourcePoints),
							  np.asarray(destinationPoints))
	return H


