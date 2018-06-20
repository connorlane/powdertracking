"""
Utilities related to particle tracking

Connor Coward
19 June 2018
"""

import cv2
import numpy as np
import math

_selectedPoints = []
_currentImage = None
_getPointsWindowName = "Select Target Positions"


def getClick(event, x, y, _, __):
	"""Callback for getting the mouse click locations"""
    global _selectedPoints
    global _currentImage

    if event == cv2.EVENT_LBUTTONDOWN:
            _selectedPoints.append((float(x), float(y)))
            cv2.circle(_currentImage, (x, y), 5, (0, 255, 0))
            cv2.imshow(_getPointsWindowName, _currentImage)


def getTargetLocations(image):
	"""Gets mouse click locations on the specified image (press 'q') to 
	   finish"""
    global _selectedPoints
    global _currentImage

    _selectedPoints = []
    _currentImage = image
	

    cv2.namedWindow(_getPointsWindowName)
    cv2.setMouseCallback(_getPointsWindowName, getClick)

    while True:
            cv2.imshow(_getPointsWindowName, image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):
                break

    return _selectedPoints


def grabframe(cap):
	"""Grab a single grayscale frame from video capture <cap>"""
    ret, frameRaw = cap.read()

    frame = None

    if ret is True:
        frame = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255
        frame = cv2.bilateralFilter(frame, 3, 25, 25)

    return ret, frame


def scaleOneSided(img):
	"""Normalize using a one-tailed distribution"""
    mean, stddev = cv2.meanStdDev(img)
    img = 255 * (img - mean) / (stddev * 12)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def euclideanDistance(p1, p2):
	"""Calculate distance between p1 and p2"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def findClosest(x, points):
	"""Find the closest point in <points> to x"""
    closestPoint = points[0]
    closestDistance = euclideanDistance(points[0], x)
    for p in points[1:]:
        distance = euclideanDistance(p, x)
        if distance < closestDistance:
            closestDistance = distance
            closestPoint = p
    return closestPoint, closestDistance


def getMedianImage(cap, maxNumFrames):
	"""Calculate the mediam image using <maxNumFrames> from 
	   video capture <cap>"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pixels = np.empty((height, width, 100))

    # Go back to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frameindex = 0
    while cap.isOpened() and frameindex < maxNumFrames:
        ret, frame = grabframe(cap)

        if ret is True:
            pixels[:, :, frameindex] = frame
        else:
            break

        # Keep track of a frame index
        frameindex = frameindex + 1

    medianImage = np.median(pixels, axis=2)
    return medianImage



