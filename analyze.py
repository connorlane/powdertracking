#! /usr/bin/env python

import sys
import math
import scipy.stats
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2


def grabframe(cap):
    ret, frameRaw = cap.read()

    frame = None

    if ret is True:
        frame = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255
        frame = cv2.bilateralFilter(frame, 3, 25, 25)

    return ret, frame


def scaleOneSided(img):
    mean, stddev = cv2.meanStdDev(img)
    img = 255 * (img - mean) / (stddev * 12)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def euclideanDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def getCenters(diff):
    ret, track = cv2.threshold(diff, 128, 256, cv2.THRESH_BINARY)
    kernelOpen = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, kernelOpen)
    track = cv2.morphologyEx(track, cv2.MORPH_OPEN, kernelOpen)

    trackRGB = cv2.cvtColor(track, cv2.COLOR_GRAY2RGB)
    trackRGB = cv2.resize(trackRGB, (newWidth, newHeight))
    blobsVid.write(trackRGB)
    cv2.imshow('blobs', trackRGB)

    _, contours, _ = cv2.findContours(track,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        if cv2.contourArea(c) >= 1:
            m = cv2.moments(c)
            center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
            centers.append((center[0], center[1]))

    return centers
_selectedPoints = []
_currentImage = None


# Callback for getting the mouse click locations
def getClick(event, x, y, _, _):
    global _selectedPoints
    global _currentImage

    if event == cv2.EVENT_LBUTTONDOWN:
            _selectedPoints.append((float(x), float(y)))
            cv2.circle(_currentImage, (x, y), 5, (0, 255, 0))
            cv2.imshow("image", _currentImage)


# Gets mouse click locations on the specified image (press 'q') to finish
def getPoints(image):
    global _selectedPoints
    global _currentImage

    _selectedPoints = []
    _currentImage = image

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", getClick)

    while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    return _selectedPoints


def findClosest(x, points):
    closestPoint = points[0]
    closestDistance = euclideanDistance(points[0], x)
    for p in points[1:]:
        distance = euclideanDistance(p, x)
        if distance < closestDistance:
            closestDistance = distance
            closestPoint = p
    return closestPoint, closestDistance


def getMedianImage(cap, maxNumFrames):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pixels = np.empty((height, width, 100))

    g_frameIndex = 0
    while cap.isOpened() and g_frameIndex < maxNumFrames:
        ret, frame = grabframe(cap)

        if ret is True:
            pixels[:, :, g_frameIndex] = frame
        else:
            break

        # Keep track of a frame index
        g_frameIndex = g_frameIndex + 1

    medianImage = np.median(pixels, axis=2)
    return medianImage


def getParticlePoints(cap, medianImage):
    particlePoints = dict()

    # Go back to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    g_frameIndex = 0
    while cap.isOpened():
        if g_frameIndex % 10 == 0:
            print "FRAME: ",  g_frameIndex

        ret, frame = grabframe(cap)

        if ret is True:
            diff = np.subtract(frame, medianImage)

            scaled = scaleOneSided(diff)

            centers = getCenters(scaled)

            if centers:
                particlePoints[g_frameIndex] = centers

            cv2.imshow('scaled', scaled)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # Keep track of a frame index
        g_frameIndex = g_frameIndex + 1

    return particlePoints

cap = cv2.VideoCapture(sys.argv[1])
ret, prevFrame = grabframe(cap)
shape = prevFrame.shape[:2]

numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

scalingRatio = 600 / float(size[0])
newWidth = int(scalingRatio * size[0])
newHeight = int(scalingRatio * size[1])

blobsVid = cv2.VideoWriter('blobs.avi',
                           cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                           50,
                           (newWidth, newHeight))
linesVid = cv2.VideoWriter('lines.avi',
                           cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                           50,
                           (newWidth, newHeight))

g_frameIndex = 0

medianImage = getMedianImage(cap, 100)
cv2.imshow('median', medianImage)

sourcePoints = getPoints(medianImage.copy())

INVALID_COORDS_MESSAGE = "Invalid Coordinates. Input as x, y"
destinationPoints = []
print "You selected", len(sourcePoints), \
    "points. Please enter the corresponding physical coordinates: "

for p in sourcePoints:
    coords = []
    while len(coords) != 2:
        try:
            coords = input(str(p) + ": ")
            if len(coords) != 2:
                print INVALID_COORDS_MESSAGE
            else:
                destinationPoints.append(coords)
        except NameError:
            print INVALID_COORDS_MESSAGE

H, _ = cv2.findHomography(np.asarray(sourcePoints),
                          np.asarray(destinationPoints))

print "Here's the calculated homography matrix:", H

particlePoints = getParticlePoints(cap, medianImage)

cv2.destroyAllWindows()

touchedPoints = dict()
segments = []
touchedPoints = dict()

for t1 in particlePoints.keys():
    if t1 not in particlePoints:
        "CONTINUING"
        continue

    points = particlePoints[t1]

    if t1 % 10 == 0:
        print "Checking frame", t1

    for p1 in points:
        t2 = t1 + 1
        while t2 not in particlePoints and t2 < numFrames:
            t2 = t2 + 1

        if t2 >= numFrames:
            continue

        p2, _ = findClosest(p1, particlePoints[t2])

        dT = t2 - t1
        dX = p2[0] - p1[0]
        dY = p2[1] - p1[1]

        mX = dX / dT
        mY = dY / dT
        bX = p1[0] - mX*t1
        bY = p1[1] - mY*t1

        for _ in xrange(1, 50):
            debugImage = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
            inlierCount = 0
            inliers = []
            t = t2 + 1
            x = bX + t * mX
            y = bY + t * mY

            skipCounter = 0
            while (x > 0 and x < shape[1] and y > 0 and y < shape[0] and
                   t < numFrames and skipCounter < 7):
                if t in particlePoints:
                    cv2.circle(debugImage, (int(x), int(y)), 4, (255, 0, 0))
                    closest, distance = findClosest((x, y), particlePoints[t])
                    if distance < 7:
                        inliers.append((t, closest[0], closest[1]))
                        inlierCount = inlierCount + 1
                        skipCounter = 0
                        cv2.circle(debugImage, closest, 4, (0, 255, 0))
                    else:
                        skipCounter = skipCounter + 1
                        cv2.circle(debugImage, closest, 4, (0, 0, 255))
                t = t + 1
                x = bX + t * mX
                y = bY + t * mY

            t = t1 - 1
            x = bX + t * mX
            y = bY + t * mY

            skipCounter = 0
            while (x > 0 and x < shape[1] and y > 0 and y < shape[0] and
                   t > 0 and skipCounter < 7):
                if t in particlePoints:
                    cv2.circle(debugImage, (int(x), int(y)), 4, (255, 0, 0))
                    closest, distance = findClosest((x, y), particlePoints[t])
                    if distance < 7:
                        inliers.append((t, closest[0], closest[1]))
                        inlierCount = inlierCount + 1
                        skipCounter = 0
                        cv2.circle(debugImage, closest, 4, (0, 255, 0))
                    else:
                        skipCounter = skipCounter + 1
                        cv2.circle(debugImage, closest, 4, (0, 0, 255))

                t = t - 1
                x = bX + t * mX
                y = bY + t * mY

            if not inliers:
                break

            mX, bX, _, _, _ = scipy.stats.linregress([(inlier[0], inlier[1]) for inlier in inliers])
            mY, bY, _, _, _ = scipy.stats.linregress([(inlier[0], inlier[2]) for inlier in inliers])

        if len(inliers) > 3:
            for t, x, y in inliers:
                if t in particlePoints:
                    particlePoints[t].remove((x, y))
                    if not particlePoints[t]:
                        del particlePoints[t]
                if t in touchedPoints:
                    touchedPoints[t].append((x, y))
                else:
                    touchedPoints[t] = [(x, y)]

            tMin = np.amin(inliers, axis=0)[0]
            tMax = np.amax(inliers, axis=0)[0]

            t = tMax + 1
            x = bX + t * mX
            y = bY + t * mY

            skipCounter = 0
            while (x > 0 and x < shape[1] and y > 0 and y < shape[0] and
                   t < numFrames and skipCounter < 3):
                if t in touchedPoints:
                    closest, distance = findClosest((x, y), touchedPoints[t])
                    if distance < 3:
                        inliers.append((t, closest[0], closest[1]))
                        tMax = t
                        skipCounter = 0
                        cv2.circle(debugImage, closest, 4, (255, 255, 255))
                    else:
                        skipCounter = skipCounter + 1

                t = t + 1
                x = bX + t * mX
                y = bY + t * mY

            t = tMin - 1
            x = bX + t * mX
            y = bY + t * mY

            skipCounter = 0
            while (x > 0 and x < shape[1] and y > 0 and y < shape[0] and
                   t > 0 and skipCounter < 3):
                if t in touchedPoints:
                    closest, distance = findClosest((x, y), touchedPoints[t])
                    if distance < 3:
                        inliers.append((t, closest[0], closest[1]))
                        tMin = t
                        skipCounter = 0
                        cv2.circle(debugImage, closest, 4, (255, 255, 255))
                    else:
                        skipCounter = skipCounter + 1

                t = t - 1
                x = bX + t * mX
                y = bY + t * mY

            segments.append((mX, bX, mY, bY, tMin, tMax))

        cv2.imshow('debug', debugImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

segmentIdCounter = 0
segmentIdMap = dict()
collisions = []
for i in xrange(0, len(segments)):
    segment1 = segments[i]
    if segment1 not in segmentIdMap.keys():
        segmentIdMap[segment1] = segmentIdCounter
        segmentIdCounter = segmentIdCounter + 1

    bestDistance = float('inf')
    bestMatch = None
    bestSegment = None

    searchFrameMin = max(i - 300, 0)
    searchFrameMax = min(i + 300, len(segments))
    for segment2 in segments[searchFrameMin:searchFrameMax]:
        if segment2 not in segmentIdMap.keys():
            segmentIdMap[segment2] = segmentIdCounter
            segmentIdCounter = segmentIdCounter + 1

        if segment2 == segment1:
            continue

        tBeg1 = segment1[4]
        tEnd1 = segment1[5]
        tBeg2 = segment2[4]
        tEnd2 = segment2[5]

        xBeg1 = segment1[0] * tBeg1 + segment1[1]
        yBeg1 = segment1[2] * tBeg1 + segment1[3]
        xEnd1 = segment1[0] * tEnd1 + segment1[1]
        yEnd1 = segment1[2] * tEnd1 + segment1[3]

        xBeg2 = segment2[0] * tBeg2 + segment2[1]
        yBeg2 = segment2[2] * tBeg2 + segment2[3]
        xEnd2 = segment2[0] * tEnd2 + segment2[1]
        yEnd2 = segment2[2] * tEnd2 + segment2[3]

        v1 = (xEnd1 - xBeg1, yEnd1 - yBeg1)
        v2 = (xEnd2 - xBeg2, yEnd2 - yBeg2)

        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            continue

        distance = euclideanDistance((xBeg2, yBeg2), (xEnd1, yEnd1))

        if abs(tEnd1 - tBeg2) < 8 and distance < 45:

            tCollision = int((tBeg2 + tEnd1) / 2)
            xCollision = int((xBeg2 + xEnd1) / 2)
            yCollision = int((yBeg2 + yEnd1) / 2)

            cosTheta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)

            v1_hat = (v1[1] / mag1)

            if cosTheta < 0.95 and v1_hat > 0.0:
                if distance < bestDistance:
                    bestDistance = distance
                    bestMatch = (tCollision, xCollision, yCollision)
                    bestSegment = segment2

    if bestMatch:
        collisions.append(bestMatch)
        print ("Matched Ends!", bestMatch, "Segments:", segmentIdMap[segment1],
               segmentIdMap[bestSegment])

cv2.destroyAllWindows()

# Go back to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

g_frameIndex = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret is True:
        frame = cv2.resize(frame, (newWidth, newHeight))
        for segment in segments:
            if (g_frameIndex > segment[4]) and (g_frameIndex < segment[5]):
                t1 = segment[4]
                x1 = int(scalingRatio * (segment[0] * segment[4] + segment[1]))
                y1 = int(scalingRatio * (segment[2] * segment[4] + segment[3]))
                x2 = int(scalingRatio * (segment[0] * g_frameIndex + segment[1]))
                y2 = int(scalingRatio * (segment[2] * g_frameIndex + segment[3]))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame,
                            str(segmentIdMap[segment]),
                            (x2 + 2, y2 + 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255))

        for t, x, y in collisions:
            if t > g_frameIndex - 5 and t < g_frameIndex + 5:
                x_scaled = int(x * scalingRatio)
                y_scaled = int(y * scalingRatio)
                cv2.circle(frame, (x_scaled, y_scaled), 4, (0, 0, 255))

        cv2.imshow('segments', frame)
        linesVid.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    # Keep track of a frame index
    g_frameIndex = g_frameIndex + 1

cap.release()
blobsVid.release()
linesVid.release()

cv2.destroyAllWindows()

# Warp the coordinates
powderPoints_warped = []
for p in collisions:
    p_warped = np.dot(H, [[p[1]], [p[2]], [1]])
    p_warped = p_warped / p_warped[2]
    powderPoints_warped.append((p[0], p_warped[0][0], p_warped[1][0]))

centroid = np.sum(powderPoints_warped, axis=0) / float(len(powderPoints_warped))

f = open('points.csv', 'w')

for p in powderPoints_warped:
    t = p[0]
    x = p[1] - centroid[1]
    y = p[2] - centroid[2]
    f.write(str(t) + ', ' + str(x) + ', ' + str(y) + '\n')
    plt.scatter(x, y)

f.close()

