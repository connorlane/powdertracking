#! /usr/bin/env python

import sys
import math
import random
import scipy.stats
import copy
import matplotlib.pyplot as plt

import numpy as np
import cv2

def normalize(img):
    mean, stddev = cv2.meanStdDev(img) 
    return (img - mean) / stddev 

def grabframe(cap):
    ret, frameRaw = cap.read()

    frame = None

    if ret == True:
        frame = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255
        #frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.bilateralFilter(frame, 3, 25, 25)

    return ret, frame

def scale(img):
    mean, stddev = cv2.meanStdDev(img)
    img = img + stddev * 12
    img = 255 * img / (20 * stddev)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def scaleOneSided(img):
    mean, stddev = cv2.meanStdDev(img)
    img = 255 * (img - mean) / (stddev * 12)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def euclideanDistance(p1, p2):
    #return np.linalg.norm(np.subtract(p1, p2))
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class trajectory:
    def __init__(self, points = []):
        self.points = points
        self.counter = 0

    def draw(self, image):
        prevCenter = self.points[0][:2]
        for center in self.points[1:]:
            cv2.line(image, prevCenter[:2], center[:2], (255, 0, 255), 1)
            prevCenter = center

    def movement(self):
        return ((self.points[-1][0] - self.points[-2][0])**2 + (self.points[-1][1] - self.points[-2][1])**2)**0.5

    def distance(self):
        prevPoint = self.points[0]
        distance = 0
        for point in self.points[1:]:
            distance = distance + euclideanDistance(prevPoint, point)
            prevPoint = point
        return distance

    def splitLinear(self):
        newTrajectories = []
        failedToFind = False
        originalSize = len(self.points)
        while len(self.points) >= 2:
            # Find a linear section
            bestNumInliers = 0
            bestInlierSet = []
            for _ in xrange(250):
                p1, p2 = random.sample(self.points, 2)
                if p1 == p2:
                    continue

                inlierSet = []
                if p1[0] == p2[0]:
                    for p in self.points:
                        d = math.fabs(p[0] - p1[0])
                        if d < 2.5:
                            inlierSet.append(p)

                else:
                    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
                    b = p1[1] - m * p1[0]
                    for p in self.points:
                        d = math.fabs(m*p[0] - p[1] + b) / math.sqrt(m*m + 1)
                        if d < 2.5:
                            inlierSet.append(p)

                if len(inlierSet) > bestNumInliers:
                    bestNumInliers = len(inlierSet)
                    bestInlierSet = inlierSet

            # Iteratively refine the linear section
            for _ in range(25):
                if len(inlierSet) < 3:
                    break

                # Calculate a best-fit line based on the inlier set
                m, b, r, p, e = scipy.stats.linregress([point[:2] for point in inlierSet])
                inlierSet = []

                for p in self.points:
                    # Calculate distance to the linear regression
                    d = math.fabs(m*p[0] - p[1] + b) / math.sqrt(m*m + 1)

                    # If inlier, add to the inlier set
                    if d < 2.5:
                        inlierSet.append(p)

            if bestNumInliers < len(inlierSet):
                bestInlierSet = inlierSet
                bestNumInliers = len(inlierSet)

            for p in bestInlierSet:
                self.points.remove(p)

            if bestNumInliers >= 4:
                # Add the points as a new linear trajectory
                newTrajectories.append(trajectory(bestInlierSet))

        # Return the list of split linear trajectories
        return newTrajectories

    def averageVelocity(self):
        return distance / (len(self.points) - 1)

    def getCost(self, c):
        dX = (self.points[-1][0] - self.points[-2][0])
        dY = (self.points[-1][1] - self.points[-2][1])

        targetX = int(round(self.points[-1][0] + dX))
        targetY = int(round(self.points[-1][1] + dY))

        #cv2.circle(disp, self.points[-2][:2], 4, (255, 0, 0))
        #cv2.circle(disp, self.points[-1][:2], 4, (0, 255, 0))
        #cv2.circle(disp, (targetX, targetY), 4, (0, 0, 255))

        cost = ((c[0] - targetX)**2 + (c[1] - targetY)**2)**0.5

        return cost

def getCenters(diff):
    #Gx = cv2.Sobel(diff, -1, 1, 0)
    #Gy = cv2.Sobel(diff, -1, 0, 1)

    #Gx2 = np.square(Gx)
    #Gy2 = np.square(Gy)

    #Gmag = np.sqrt(Gx2 + Gy2)

    #diff = scaleOneSided(diff)

    ret, track = cv2.threshold(diff, 128, 256, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    kernelOpen = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, kernelOpen)
    track = cv2.morphologyEx(track, cv2.MORPH_OPEN, kernelOpen)

    trackRGB = cv2.cvtColor(track, cv2.COLOR_GRAY2RGB)
    blobsVid.write(trackRGB)
    cv2.imshow('blobs', trackRGB)

    contours, _ = cv2.findContours(track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        if cv2.contourArea(c) >= 1:
            m = cv2.moments(c)
            center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
            centers.append((center[0], center[1]))
            #print "APPENDING: ", center
            #cv2.circle(track, center, 5, (255, 255, 255))

    return centers

def matchCenters(centers, trajectories, frameIndex):
    bestTrajectories = dict()
    for c in centers:
        if trajectories:
            #print "NUMTRAJ: ", len(trajectories)
            bestCost = trajectories[0].getCost(c)
            bestTraj = trajectories[0]
            for traj in trajectories[1:]:
                cost = traj.getCost(c)
                if cost < bestCost:
                    bestTraj = traj
                    bestCost = cost
            bestTrajectories[c] = bestTraj

    for traj in trajectories:
        if centers:
            #print "NUMTRAJ: ", len(trajectories)
            bestCost = traj.getCost(centers[0])
            #print "TRAJCOST: ", bestCost
            bestCenter = centers[0]
            #print "CENTERS: ", centers
            for c in centers[1:]:
                cost = traj.getCost(c)
                #print "TRAJCOST: ", bestCost
                if cost < bestCost:
                    bestCenter = c
                    bestCost = cost
            mov = traj.movement()
            #print "BESTCOST: ", bestCost
            #print "MOVEMENT: ", mov
            if bestTrajectories[bestCenter] == traj and mov and bestCost < 2 * mov + 3:
                traj.points.append((bestCenter[0], bestCenter[1], frameIndex))
                centers.remove(bestCenter)
            else:
                traj.counter = traj.counter + 1

def matchEnds(startTrajectories, endTrajectories):
    newPowderPoints = []

    # Test all possible combinations of linear segments
    # (This is fine since there will only be a handfull of segments)
    for linearTraj1 in startTrajectories:
        for linearTraj2 in endTrajectories:
            # Skip matching to self
            if linearTraj2 == linearTraj1:
                continue

            # Get the start point of this trajectory
            start = linearTraj1.points[0]
            # Get the end points of that trajectory
            end = linearTraj2.points[-1]

            # Calculate time & physical distances
            timeDistance = start[2] - end[2]
            physicalDistance = euclideanDistance(end[:2], start[:2])

            # If matching in time and space...
            #print "TIMEDIST SAMETRAJ: ", timeDistance
            #print "SPACEDIST SAMETRAJ: ", physicalDistance
            if timeDistance >= -1 and timeDistance <= 3 and physicalDistance < 55:
                # Check for lined up trajectories
                v1 = (linearTraj1.points[-1][0] - linearTraj1.points[0][0],
                      linearTraj1.points[-1][1] - linearTraj1.points[0][1])
                v2 = (linearTraj2.points[-1][0] - linearTraj2.points[0][0],
                      linearTraj2.points[-1][1] - linearTraj2.points[0][1])

                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

                # If we're looking at a degenerate trajectory
                if mag1 == 0 or mag2 == 0:
                    continue
                    
                cosTheta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)

                if linearTraj1 in startTrajectories:
                    startTrajectories.remove(linearTraj1)

                #print "COSTHETA: ", cosTheta
                if cosTheta > 0.9:
                    #print "COMBINING SAMETRAJ"
                    # Combine the two segments
                    linearTraj2.points.extend(linearTraj1.points)
                elif cosTheta < 0.8:
                    #print "MATCH SAMERAJ"

                    if cosTheta > -0.99:
                        m1, b1, r1, p1, e1 = scipy.stats.linregress([p[:2] for p in linearTraj1.points])
                        m2, b2, r2, p2, e2 = scipy.stats.linregress([p[:2] for p in linearTraj2.points])

                        if math.isnan(m1) or math.isnan(m2):
                            x = end[0]
                            y = end[1]
                        else:
                            x = (b1 - b2) / (m2 - m1)
                            y = m1 * x + b1

                        x = int(round(x))
                        y = int(round(y))
                    else:
                        x = end[0]
                        y = end[1]

                    newPowderPoints.append((x, y, (end[2] + start[2])/2))
                    #cv2.circle(disp, (x, y), 10, (255, 0, 255))
                    endTrajectories.remove(linearTraj2)

    return newPowderPoints

def extractCollisionsAndPrune(trajectories, unmatchedSegments):
    newPowderPoints = []
    for traj in trajectories:
        if traj.counter >= 2:

            # Split the trajectories into linear segments
            splitTrajectories = traj.splitLinear()                     

            # If we have two or more linear segments...
            if len(splitTrajectories) >= 2:
                # Check for end matches within the present trajectory
                powderPoints = matchEnds(splitTrajectories, splitTrajectories)
                newPowderPoints.extend(powderPoints)

            # Check for end matches in the list of existing segments
            powderPoints = matchEnds(splitTrajectories, unmatchedSegments)
            newPowderPoints.extend(powderPoints)
          
            # Add the remaining linear trajectories to the unmatched segments list
            unmatchedSegments.extend(splitTrajectories)

            # Finally, kill the trajectory
            trajectories.remove(traj)

    return newPowderPoints

def findNewTrajectories(centers, prevCenters):
    newTrajectories = []

    bestCenters = dict()
    for center in centers:
        if prevCenters:
            bestDistance = ((center[0] - prevCenters[0][0])**2 + (center[1] - prevCenters[0][1])**2)**0.5
            bestCenter = prevCenters[0]
            for prevCenter in prevCenters:
                distance = ((center[0] - prevCenter[0])**2 + (center[1] - prevCenter[1])**2)**0.5
                if distance < bestDistance:
                    bestDistance = distance
                    bestCenter = prevCenter
            bestCenters[center] = bestCenter

    prevBestCenters = dict()
    for prevCenter in prevCenters:
        if centers:
            bestDistance = ((centers[0][0] - prevCenter[0])**2 + (centers[0][1] - prevCenter[1])**2)**0.5
            bestCenter = centers[0]
            for center in centers:
                distance = ((center[0] - prevCenter[0])**2 + (center[1] - prevCenter[1])**2)**0.5
                if distance < bestDistance:
                    bestDistance = distance
                    bestCenter = center
            prevBestCenters[prevCenter] = bestCenter

    matches = dict()
    for center in bestCenters:
        prevCenter = bestCenters[center]
        if prevBestCenters[prevCenter] == center:
            matches[center] = prevCenter

    for center in matches:
        prevCenter = matches[center]
        #print "NEW TRAJECTORY"
        newTrajectories.append(trajectory([prevCenter, center]))
        centers.remove(center)

    return newTrajectories

_selectedPoints = []
_currentImage = None

# Callback for getting the mouse click locations
def getClick(event, x, y, flags, param):
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
                break;

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
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    pixels = np.empty((height, width, 100))

    g_frameIndex = 0
    while cap.isOpened() and g_frameIndex < maxNumFrames:
        ret, frame = grabframe(cap)

        if ret == True:
            pixels[:,:,g_frameIndex] = frame
        else:
            break

        # Keep track of a frame index
        g_frameIndex = g_frameIndex + 1

    medianImage = np.median(pixels, axis=2)
    return medianImage

def getParticlePoints(cap, medianImage):
    particlePoints = dict()

    # Go back to first frame
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
 
    g_frameIndex = 0
    while cap.isOpened():
        print "FRAME: ",  g_frameIndex
 
        ret, frame = grabframe(cap)
 
        if ret == True:
            diff = np.subtract(frame, medianImage)
 
            scaled = scaleOneSided(diff)
 
            centers = getCenters(scaled)

            #particlePoints.append(centers)
            
            if centers:
                particlePoints[g_frameIndex] = centers
                #particlePoints.extend([(g_frameIndex, center[0], center[1]) for center in centers])
 
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
prevTracks = [np.zeros(prevFrame.shape[:2])] * 10
prevTrack = np.zeros(prevFrame.shape[:2])
prevPrevTrack = np.zeros(prevFrame.shape[:2])
prevPrevPrevTrack = np.zeros(prevFrame.shape[:2])

numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

blobsVid = cv2.VideoWriter('blobs.avi', cv2.cv.FOURCC('X','V','I','D'), 60, size)
linesVid = cv2.VideoWriter('lines.avi', cv2.cv.FOURCC('X','V','I','D'), 60, size)

g_trajectories = []
g_misfits = []
g_prevCenters = []
g_unmatchedSegments = []
g_powderPoints = []
g_frameIndex = 0
g_centers = dict()

medianImage = getMedianImage(cap, 100)
cv2.imshow('median', medianImage)

sourcePoints = getPoints(medianImage.copy())

INVALID_COORDS_MESSAGE = "Invalid Coordinates. Input as x, y"
destinationPoints = []
print "You selected", len(sourcePoints), "points. Please enter the corresponding physical coordinates: "
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

H, mask = cv2.findHomography(np.asarray(sourcePoints), np.asarray(destinationPoints))

print "Here's the calculated homography matrix:", H

particlePoints = getParticlePoints(cap, medianImage)

cv2.destroyAllWindows()

touchedPoints = dict()
segments = []
touchedPoints = dict()

for t1 in particlePoints.keys():
    if not t1 in particlePoints:
        "CONTINUING"
        continue

    points = particlePoints[t1]
    print "Checking frame", t1
    for p1 in points:
        t2 = t1 + 1
        while not t2 in particlePoints and t2 < numFrames:
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
            while x > 0 and x < shape[1] and y > 0 and y < shape[0] and t < numFrames and skipCounter < 7:
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
            while x > 0 and x < shape[1] and y > 0 and y < shape[0] and t > 0 and skipCounter < 7:
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

            mX, bX, rX, pX, eX = scipy.stats.linregress([(inlier[0], inlier[1]) for inlier in inliers])
            mY, bY, rY, pY, eY = scipy.stats.linregress([(inlier[0], inlier[2]) for inlier in inliers])

        if len(inliers) > 4:
            print "Segment Found!"
            for t, x, y in inliers:
                if t in particlePoints:
                    particlePoints[t].remove((x,y))
                    if not particlePoints[t]:
                        del particlePoints[t]
                if t in touchedPoints:
                    touchedPoints[t].append((x,y))
                else:
                    touchedPoints[t] = [(x,y)]

            tMin = np.amin(inliers, axis=0)[0]
            tMax = np.amax(inliers, axis=0)[0]
        
#            t = tMax + 1
#            x = bX + t * mX 
#            y = bY + t * mY 
#
#            skipCounter = 0
#            while x > 0 and x < shape[1] and y > 0 and y < shape[0] and t < numFrames and skipCounter < 3:
#                if t in touchedPoints:
#                    closest, distance = findClosest((x, y), touchedPoints[t])
#                    if distance < 3:
#                        print "Found Overlapping inlier +!"
#                        inliers.append((t, closest[0], closest[1]))
#                        tMax = t
#                        skipCounter = 0
#                        cv2.circle(debugImage, closest, 4, (255, 255, 255))
#                    else:
#                        skipCounter = skipCounter + 1
#
#                t = t + 1
#                x = bX + t * mX 
#                y = bY + t * mY 
#
#            t = tMin - 1
#            x = bX + t * mX 
#            y = bY + t * mY 
#
#            skipCounter = 0
#            while x > 0 and x < shape[1] and y > 0 and y < shape[0] and t > 0 and skipCounter < 3:
#                if t in touchedPoints:
#                    closest, distance = findClosest((x, y), touchedPoints[t])
#                    if distance < 3:
#                        print "Found Overlapping inlier -!"
#                        inliers.append((t, closest[0], closest[1]))
#                        tMin = t
#                        skipCounter = 0
#                        cv2.circle(debugImage, closest, 4, (255, 255, 255))
#                    else:
#                        skipCounter = skipCounter + 1
#
#                t = t - 1
#                x = bX + t * mX 
#                y = bY + t * mY 
#
            segments.append((mX, bX, mY, bY, tMin, tMax))

        cv2.imshow('debug', debugImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

collisions = []
for segment1 in segments:
    for segment2 in segments:
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

        if abs(tEnd1 - tBeg2) < 4 and distance < 25:
            tCollision = int((tBeg2 + tEnd1) / 2)
            xCollision = int((xBeg2 + xEnd1) / 2)
            yCollision = int((yBeg2 + yEnd1) / 2)          

            cosTheta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2) 

            v1_hat = (v1[1] / mag1)

            print "cosTheta:",cosTheta
            print "v1_hat:",v1_hat

            if cosTheta < 0.8 and v1_hat > 0.1:
                collisions.append((tCollision, xCollision, yCollision))
                print "Matched Ends!",(tCollision, xCollision, yCollision)
                

cv2.destroyAllWindows()

# Go back to first frame
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

g_frameIndex = 0
while cap.isOpened():
    print "FRAME: ",  g_frameIndex

    ret, frame = cap.read()

    if ret == True:
        for segment in segments:
            if (g_frameIndex > segment[4]) and (g_frameIndex < segment[5]):
                print "SEGMENT:",segment
                t1 = segment[4]
                x1 = int(segment[0] * segment[4] + segment[1])
                y1 = int(segment[2] * segment[4] + segment[3])
                x2 = int(segment[0] * g_frameIndex + segment[1])
                y2 = int(segment[2] * g_frameIndex + segment[3])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

        for t, x, y in collisions:
            if t > g_frameIndex - 2 and t < g_frameIndex + 2:
                cv2.circle(frame, (x, y), 3, (255, 255, 0))

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
    p_warped = np.dot(H, [ [p[1]], [p[2]], [1] ])
    p_warped = p_warped / p_warped[2]
    powderPoints_warped.append((p[0], p_warped[0][0], p_warped[1][0]))

centroid = np.sum(powderPoints_warped, axis=0) / float(len(powderPoints_warped))

f = open('points.csv', 'w')

for p in powderPoints_warped:
    t = p[0]
    x = p[1] - centroid[1]
    y = p[2] - centroid[2]
    f.write(str(t) + ', ' + str(y) + ', ' + str(y) + '\n')
    plt.scatter(x, y)

f.close()

plt.show()

