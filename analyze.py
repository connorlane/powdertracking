#! /usr/bin/env python

import sys
import math
import scipy.stats
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2

import util
import homography
import track
import visualize


# Create a video capture object from the given file
cap = cv2.VideoCapture(sys.argv[1])

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
blobsVid = cv2.VideoWriter('blobs.avi',
                           cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                           50,
                           (newWidth, newHeight))

# Calculate the median image from the first few frames. 
#   This effectively gets rid of the powder particles.
medianImage = util.getMedianImage(cap, 100)

# Get the image coordinates of the target points in the image
print "Select 4 or more target points in the median image, then press 'Q'"
sourcePoints = homography.getTargetLocations(medianImage.copy())

# Get the corresponding physical locations of the target points
print "You selected", len(sourcePoints), \
	"points. Please enter the corresponding physical coordinates: "
destinationPoints = homography.getPhysicalCoordinates(sourcePoints)

# Calculate a homography matrix to translate between the source and target pts
H = homography.getHomographyMatrix(sourcePoints, destinationPoints)
print "Here's the calculated homography matrix:", H

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

# Play back results
visualize.playbackResults(cap, collisions, segments, segmentIdMap, 1.75)

# Release capture and video writer objects
cap.release()
blobsVid.release()

# Clean up windows
cv2.destroyAllWindows()

# Warp the coordinates to real-world coordinates
powderPoints_warped = []
for p in collisions:
    p_warped = np.dot(H, [[p[1]], [p[2]], [1]])
    p_warped = p_warped / p_warped[2]
    powderPoints_warped.append((p[0], p_warped[0][0], p_warped[1][0]))

# Find the centroid
centroid = np.sum(powderPoints_warped, axis=0) / float(len(powderPoints_warped))

# Create a CSV file for writing the collision locations
f = open('points.csv', 'w')
for p in powderPoints_warped:
    t = p[0]
    x = p[1] - centroid[1]
    y = p[2] - centroid[2]
    f.write(str(t) + ', ' + str(x) + ', ' + str(y) + '\n')
    plt.scatter(x, y)

f.close()

