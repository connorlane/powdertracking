
from collections import Iterable
import numpy as np
import cv2
import os.path
import sys

import trackingEngine.util

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


def interactiveHomographySelection(videoFile):
	# Create a video capture object from the given file
	video = cv2.VideoCapture(videoFile)

	# Calculate the median image from the first few frames. 
	#   This effectively gets rid of the powder particles.
	medianImage = trackingEngine.util.getMedianImage(video, 100)

	# Get the image coordinates of the target points in the image
	print "Select 4 or more target points in the median image, then press 'Q'"
	sourcePoints = getTargetLocations(medianImage)

	# Get the corresponding physical locations of the target points
	print "You selected", len(sourcePoints), \
		"points. Please enter the corresponding physical coordinates: "
	destinationPoints = getPhysicalCoordinates(sourcePoints)

	# Calculate a homography matrix to translate between the source and target pts
	H = getHomographyMatrix(sourcePoints, destinationPoints)
	print "Here's the calculated homography matrix:", H

	return H


def save(H, homographyFile):	
	# Save the matrix to file
	with open(homographyFile, "wb") as outfile:
		np.savetxt(outfile, H)


def load(homographyFile):
	return np.loadtxt(homographyFile)


def printUsage():
	print "Usage:", sys.argv[0], "<video file>"


if __name__ == "__main__":
	if len(sys.argv) != 2:
		printUsage()	
		sys.exit(-1)

	videoFile = sys.argv[1]

	if not os.path.isfile(videoFile):
		print "Error reading file"
		printUsage()
		sys.exit(-1)

	fileNameOnly = os.path.basename(videoFile)
	baseName = os.path.splitext(fileNameOnly)[0]

	homographyFile = baseName + ".hom"

	H = interactiveHomographySelection(videoFile, homographyFile)

	save(H)


