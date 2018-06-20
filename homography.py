"""
Generates homography matrices to translate pixel coordinates to real-world 
   coordinates. This script prompts the user to select known target locations 
   in an image and then calculates a transformation matrix to convert video 
   pixel coordinates to physical coordinates.

Connor Coward
19 June 2018
"""

from collections import Iterable
import numpy as np
import cv2
import os.path
import sys

import trackingEngine.util

def getPhysicalCoordinates(sourcePoints):
	"""Collects physical coordinates from user using input()"""
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
	"""Calculate homography matrix given 4 or more source and destination 
	   points"""
	H, _ = cv2.findHomography(np.asarray(sourcePoints),
							  np.asarray(destinationPoints))
	return H


def interactiveHomographySelection(videoFile):
	"""Prompts the user to visually select on each target point in the 
	   video scene."""
	# Create a video capture object from the given file
	video = cv2.VideoCapture(videoFile)

	# Calculate the median image from the first few frames. 
	#   This effectively gets rid of the powder particles.
	medianImage = trackingEngine.util.getMedianImage(video, 100)

	# Get the image coordinates of the target points in the image
	print "Select 4 or more target points in the median image, then press 'Q'"
	sourcePoints = trackingEngine.util.getTargetLocations(medianImage)

	# Get the corresponding physical locations of the target points
	print "You selected", len(sourcePoints), \
		"points. Please enter the corresponding physical coordinates: "
	destinationPoints = getPhysicalCoordinates(sourcePoints)

	# Calculate a homography matrix to translate between the source 
	#     and target points
	H = getHomographyMatrix(sourcePoints, destinationPoints)
	print "Here's the calculated homography matrix:", H

	return H


def save(H, homographyFile):	
	"""Saves a homography matrix to file"""
	# Save the matrix to file
	with open(homographyFile, "wb") as outfile:
		np.savetxt(outfile, H)


def load(homographyFile):
	"""Loads a homography matrix from file"""
	return np.loadtxt(homographyFile)


def printUsage():
	"""Helper function to print usage for this script"""
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

	H = interactiveHomographySelection(videoFile)

	save(H, homographyFile)


