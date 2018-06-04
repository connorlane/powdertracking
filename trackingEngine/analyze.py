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

def run(videoFile):
	# Get the base name of the video file. We'll use this for naming data files
	fileNameOnly = os.path.basename(videoFile)
	baseName = os.path.splitext(fileNameOnly)[0]

	# Create a video capture object from the given file
	cap = cv2.VideoCapture(videoFile)

	# Calculate the median image from the first few frames. 
	#   This effectively gets rid of the powder particles.
	medianImage = util.getMedianImage(cap, 100)

	# Grab one frame and find the shape of the frame/image
	ret, prevFrame = util.grabframe(cap)
	shape = prevFrame.shape[:2]

	# Calculate some other basic parameters about the video
	numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

	# Scale the video to a fixed height
	scalingRatio = 600 / float(size[0])
	newWidth = int(scalingRatio * size[0])
	newHeight = int(scalingRatio * size[1])

	# Create video writers for result videos
	blobsVid = cv2.VideoWriter(baseName + '_blobs.avi',
							   cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
							   50,
							   size)

	# Walk through each frame and get a list of potential particle
	#   locations for each frame
	particlePoints = track.getParticlePoints(cap, medianImage)

	# Get segments (particle trajectories) from the noisy list of 
	#   potential particle locations
	segments, segmentIdMap = track.getSegmentsFromPoints(particlePoints, shape, numFrames)

	# Calculate collisions from the segments. Collisions are locations
	#   where one segment ends exactly where another begins.
	#   This must be true both spatially and temporally.
	collisions = track.getCollisionsFromSegments(segments, segmentIdMap)

	# Release capture and video writer objects
	cap.release()
	blobsVid.release()

	# Clean up windows
	cv2.destroyAllWindows()

	return collisions, segments, segmentIdMap


def generateCsvFile(collisions, H, csvFileName):
	# Warp the coordinates to real-world coordinates
	powderPoints_warped = []
	for p in collisions:
		p_warped = np.dot(H, [[p[1]], [p[2]], [1]])
		p_warped = p_warped / p_warped[2]
		powderPoints_warped.append((p[0], p_warped[0][0], p_warped[1][0]))

	# Find the centroid
	centroid = np.sum(powderPoints_warped, axis=0) / float(len(powderPoints_warped))

	# Create a CSV file for writing the collision locations
	f = open(csvFileName, "w")
	for p in powderPoints_warped:
		t = p[0]
		x = p[1] - centroid[1]
		y = p[2] - centroid[2]
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
		print "Usage:", sys.argv[0], "<video file>"
		print "note: generate a homography file with the homography.py utility"
		print "      homography file should have the same name as video file, "
		print "      but with a .hom extension"

	if len(sys.argv) != 2:
		printUsage()	
		sys.exit(-1)

	videoFile = sys.argv[1]

	if not os.path.isfile(videoFile):
		print "Error reading file"
		printUsage()
		sys.exit(-1)

	# Get the base name of the video file. We'll use this for naming data files
	fileNameOnly = os.path.basename(videoFile)
	baseName = os.path.splitext(fileNameOnly)[0]

	homographyFile = baseName + ".hom"

	if not os.path.isfile(homographyFile):
		print "Error reading homography file"
		printUsage()
		sys.exit(-1)

	pickleFile = baseName + ".p"

	collisions, segments, segmentIdMap = run(videoFile)	

	csvFileName = baseName + ".csv"

	H = np.loadtxt(homographyFile)

	generateCsvFile(collisions, H, csvFileName)

	save(collisions, segments, segmentIdMap, pickleFile)


