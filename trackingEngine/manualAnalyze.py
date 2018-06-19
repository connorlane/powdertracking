#! /usr/bin/env python

import sys
import math
import scipy.stats
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cPickle as pickle
import os.path
import util
import track
import csv
import analyze

FP_COLOR = (255, 0, 0)
FN_COLOR = (0, 0, 255)
COMPUTERGENERATED_COLOR = (0, 255, 0)

_falsePositives = dict()
_falseNegatives = dict()
_frameIndex = 0
_frame = None
_getPointsWindowName = "Select Target Positions"


# Callback for getting the mouse click locations
def getClick(event, x, y, _, __):
	global _particlePoints
	global _frameIndex

	if event == cv2.EVENT_MBUTTONDOWN:
		cv2.circle(_frame, (int(x), int(y)), 5, FP_COLOR)
		cv2.imshow(_getPointsWindowName, _frame)

		if _frameIndex in _falsePositives:
			_falsePositives[_frameIndex].append((float(x), float(y)))
		else:
			_falsePositives[_frameIndex] = [(float(x), float(y))]

	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(_frame, (int(x), int(y)), 5, FN_COLOR)
		cv2.imshow(_getPointsWindowName, _frame)

		if _frameIndex in _falseNegatives:
			_falseNegatives[_frameIndex].append((float(x), float(y)))
		else:
			_falseNegatives[_frameIndex] = [(float(x), float(y))]

	if event == cv2.EVENT_RBUTTONDOWN:
		if _frameIndex in _falsePositives:
			_falsePositives[_frameIndex] = [p for p in _falsePositives[_frameIndex] if np.linalg.norm([p[0] - float(x), p[1] - float(y)]) >= 5.0]
		if _frameIndex in _falseNegatives:
			_falseNegatives[_frameIndex] = [p for p in _falseNegatives[_frameIndex] if np.linalg.norm([p[0] - float(x), p[1] - float(y)]) >= 5.0]

def run(videoFile, collisions, segments, segmentIdMap):
	global _frame
	global _frameIndex
	global _particlePoints

	# Get the base name of the video file. We'll use this for naming data files
	fileNameOnly = os.path.basename(videoFile)
	baseName = os.path.splitext(fileNameOnly)[0]

	# Create a video capture object from the given file
	cap = cv2.VideoCapture(videoFile)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	newWidth = int(width * scale)
	newHeight = int(height * scale)

	_particlePoints = dict()
	_frameIndex = 0

	while cap.isOpened():
		cap.set(cv2.CAP_PROP_POS_FRAMES, _frameIndex)
		ret, _frame = cap.read()
		_frame = cv2.resize(_frame, (newWidth, newHeight))

		cv2.namedWindow(_getPointsWindowName)
		cv2.setMouseCallback(_getPointsWindowName, getClick)

		if ret is True:
			for segment in segments:
				if (_frameIndex > segment[4]) and (_frameIndex < segment[5]):
					t1 = segment[4]
					x1 = int(scale * (segment[0] * segment[4] + segment[1]))
					y1 = int(scale * (segment[2] * segment[4] + segment[3]))
					x2 = int(scale * (segment[0] * _frameIndex + segment[1]))
					y2 = int(scale * (segment[2] * _frameIndex + segment[3]))
					#cv2.line(_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
					cv2.putText(_frame,
								str(segmentIdMap[segment]),
								(x2 + 2, y2 + 2),
								cv2.FONT_HERSHEY_SIMPLEX,
								0.4,
								(255, 255, 255))

			for t, x, y in collisions:
				if t > _frameIndex - 3 and t < _frameIndex + 3:
					x_scaled = int(scale * x)
					y_scaled = int(scale * y)
					cv2.circle(_frame, (x_scaled, y_scaled), 4, COMPUTERGENERATED_COLOR)

			if _frameIndex in _falsePositives:
				for fp in _falsePositives[_frameIndex]:
					cv2.circle(_frame, (int(fp[0]), int(fp[1])), 4, FP_COLOR)

			if _frameIndex in _falseNegatives:
				for fp in _falseNegatives[_frameIndex]:
					cv2.circle(_frame, (int(fp[0]), int(fp[1])), 4, FN_COLOR)

			cv2.putText(_frame,
						str(_frameIndex),
						(50, 50),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.0,
						(255, 255, 255))

			cv2.imshow(_getPointsWindowName, _frame)

			quit = False
			while True:
					cv2.imshow(_getPointsWindowName, _frame)
					key = cv2.waitKey(0) & 0xFF

					if key == ord("a"):
						nextFrame = max(_frameIndex - 1, 0)
						break
					if key == ord("d"):
						nextFrame = _frameIndex + 1
						break
					if key == ord("q"):
						quit = True
						break

			_frameIndex = nextFrame

			print "FrameIndex:", _frameIndex

			if quit == True:
				break

		else:
			break
	# Release capture and video writer objects
	cap.release()

	# Clean up windows
	cv2.destroyAllWindows()

	return


def generateCsvFile(collisions, H, csvFileName):
	# Warp the coordinates to real-world coordinates
	powderPoints_warped = []
	for p in collisions:
		p_warped = np.dot(H, [[p[1]], [p[2]], [1]])
		p_warped = p_warped / p_warped[2]
		powderPoints_warped.append((p[0], p_warped[0][0], p_warped[1][0]))

	# Create a CSV file for writing the collision locations
	f = open(csvFileName, "w")
	for p in powderPoints_warped:
		t = p[0]
		x = p[1]
		y = p[2]
		f.write(str(t) + ', ' + str(x) + ', ' + str(y) + '\n')
		plt.scatter(x, y)

	f.close()


def save(collisions, segments, segmentIdMap, pickleFile):
	# Pickle the trajectories and collisions
	resultsData = {
		"collisions": collisions,
		"segments": segments,
		"segmentIdMap": segmentIdMap,
	}
	with open(pickleFile, 'w') as outfile:
		pickle.dump(resultsData, outfile)


def load(pickleFile):
	with open(pickleFile, 'r') as infile:
		rd = pickle.load(infile)

	return rd["collisions"], rd["segments"], rd["segmentIdMap"]


if __name__ == "__main__":
	def printUsage():
		print "Usage:", sys.argv[0], "<video file> <scale>"
		print "note: generate a homography file with the homography.py utility"
		print "      homography file should have the same name as video file, "
		print "      but with a .hom extension"

	if len(sys.argv) != 3:
		printUsage()	
		sys.exit(-1)

	videoFile = sys.argv[1]
	if not os.path.isfile(videoFile):
		print "Error reading file"
		printUsage()
		sys.exit(-1)

	scale = float(sys.argv[2])

	# Get the base name of the video file. We'll use this for naming data files
	fileNameOnly = os.path.basename(videoFile)
	baseName = os.path.splitext(fileNameOnly)[0]

	# Load the homography file
	homographyFile = baseName + ".hom"
	if not os.path.isfile(homographyFile):
		print "Error reading homography file"
		printUsage()
		sys.exit(-1)
	H = np.loadtxt(homographyFile)

	# Get the pickle file
	pickleFile = baseName + ".p"
	if not os.path.isfile(pickleFile):
		print "Error pickle file"
		printUsage()
		sys.exit(-1)
	collisions, segments, segmentIdMap = analyze.load(pickleFile)

	# Results stored in globals because of UI click callbacks
	run(videoFile, collisions, segments, segmentIdMap)	

	# Convert false positives to explicit list of 3-tuples from sparse dict
	falsePositives_list = []
	for t, points in _falsePositives.iteritems():
		falsePositives_list.extend([(t, p[0], p[1]) for p in points])

	# Convert false negatives
	falseNegatives_list = []
	for t, points in _falseNegatives.iteritems():
		falseNegatives_list.extend([(t, p[0], p[1]) for p in points])

	# Save the false positives & negatives
	csvFileName_falsePositives = baseName + "_falsePositives.csv"
	csvFileName_falseNegatives = baseName + "_falseNegatives.csv"
	generateCsvFile(falsePositives_list, H, csvFileName_falsePositives)
	generateCsvFile(falseNegatives_list, H, csvFileName_falseNegatives)

