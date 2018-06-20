"""A main entry point for the tracking software. Demonstrates a typical example 
   of how to use the modules in this project.

Connor Coward
19 June 2018
"""

import sys
import os.path
import cPickle as pickle

import trackingEngine
import homography
import visualize

def printUsage():
	print "Usage:", sys.argv[0], "<video file>"

if len(sys.argv) != 2:
	printUsage()	
	sys.exit(-1)

videoFile = sys.argv[1]

if not os.path.isfile(videoFile):
	print "Error reading video file"
	printUsage()
	sys.exit(-1)

# Get the base name of the video file. We'll use this for naming data files
fileNameOnly = os.path.basename(videoFile)
baseName = os.path.splitext(fileNameOnly)[0]


homographyFile = baseName + ".hom"

if os.path.exists(homographyFile):
	print "Loading existing homography file"
	H = homography.load(homographyFile)
else:
	# Get the H matrix and save
	H = homography.interactiveHomographySelection(videoFile)
	homography.save(H, homographyFile)


pickleFile = baseName + ".p"

if os.path.exists(pickleFile):
	print "Loading existing results"
	collisions, segments, segmentIdMap = trackingEngine.analyze.load(pickleFile)
else:
	collisions, segments, segmentIdMap = trackingEngine.analyze.run(videoFile)

	csvFileName = baseName + ".csv"
	trackingEngine.analyze.generateCsvFile(collisions, H, csvFileName)

	trackingEngine.analyze.save(collisions, segments, segmentIdMap, pickleFile)

# Play back the results
visualize.playbackResults(
	videoFile, collisions, segments, segmentIdMap, 2.0, 10)

