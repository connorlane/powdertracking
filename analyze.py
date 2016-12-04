#! /usr/bin/env python

import math
import random
import scipy.stats

try:
    import numpy as np
    import cv2
except:
    print "Error importing dependencies"
    exit(-1)

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
    img = 255 * img / (stddev * 14)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def euclideanDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class trajectory:
    def __init__(self, points = []):
        self.points = points
        self.counter = 0

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
            for _ in xrange(1000):
                p1, p2 = random.sample(self.points, 2)
                if p1 == p2:
                    continue

                inlierSet = []
                if p1[0] == p2[0]:
                    for p in self.points:
                        d = math.fabs(p[0] - p1[0])
                        if d < 3.5:
                            inlierSet.append(p)

                else:
                    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
                    b = p1[1] - m * p1[0]
                    for p in self.points:
                        d = math.fabs(m*p[0] - p[1] + b) / math.sqrt(m*m + 1)
                        if d < 3.5:
                            inlierSet.append(p)

                if len(inlierSet) > bestNumInliers:
                    bestNumInliers = len(inlierSet)
                    bestInlierSet = inlierSet

            print "BEST NUM INLIER: ", bestNumInliers

            # Iteratively refine the linear section
            for _ in range(100):
                if len(inlierSet) < 3:
                    print "DONK"
                    break

                # Calculate a best-fit line based on the inlier set
                m, b, r, p, e = scipy.stats.linregress([point[:2] for point in inlierSet])
                inlierSet = []

                for p in self.points:
                    # Calculate distance to the linear regression
                    d = math.fabs(m*p[0] - p[1] + b) / math.sqrt(m*m + 1)

                    # If inlier, add to the inlier set
                    if d < 3.5:
                        inlierSet.append(p)

            if bestNumInliers < len(inlierSet):
                bestInlierSet = inlierSet
                bestNumInliers = len(inlierSet)

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for p in bestInlierSet:
                self.points.remove(p)
                cv2.circle(disp, p[:2], 3, color)
            print "NEW LENGTH: ", len(bestInlierSet)

            if bestNumInliers >= 3:
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

        cv2.circle(disp, self.points[-2][:2], 4, (255, 0, 0))
        cv2.circle(disp, self.points[-1][:2], 4, (0, 255, 0))
        cv2.circle(disp, (targetX, targetY), 4, (0, 0, 255))

        cost = ((c[0] - targetX)**2 + (c[1] - targetY)**2)**0.5

        cv2.putText(disp, str(cost), self.points[-1][:2], cv2.FONT_HERSHEY_PLAIN, 0.86, (0, 0, 0))
        #print "GETCOST COST: ", cost

        return cost


cap = cv2.VideoCapture('ti.mov') 
ret, prevFrame = grabframe(cap)
prevTracks = [np.zeros(prevFrame.shape[:2])] * 10
prevTrack = np.zeros(prevFrame.shape[:2])
prevPrevTrack = np.zeros(prevFrame.shape[:2])
prevPrevPrevTrack = np.zeros(prevFrame.shape[:2])

#cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('video.avi', cv2.cv.FOURCC('X','V','I','D'), 20, size)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 256

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.05

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.2
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else: 
    detector = cv2.SimpleBlobDetector_create(params)

trajectories = []
misfits = []
prevCenters = []
unmatchedSegments = []
powderPoints = []
frameIndex = 0

#cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    print "FRAME: ",  frameIndex

    ret, frame = grabframe(cap)

    if ret == True:
        diff = np.subtract(frame, prevFrame)

        Gx = cv2.Sobel(diff, -1, 1, 0)
        Gy = cv2.Sobel(diff, -1, 0, 1)

        Gx2 = np.square(Gx)
        Gy2 = np.square(Gy)

        Gmag = np.sqrt(Gx2 + Gy2)

        diff = scaleOneSided(Gmag)

        ret, track = cv2.threshold(diff, 100, 256, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        kernelOpen = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, kernel)
        track = cv2.morphologyEx(track, cv2.MORPH_OPEN, kernel)
        diff = np.maximum(diff, track)

        contours, heirarchy = cv2.findContours(track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #disp = cv2.max(0.8 * track / 255, 0.4 * prevTrack / 255)
        #disp = cv2.max(disp, 0.2 * prevPrevTrack / 255)
        #disp = cv2.max(disp, 0.1 * prevPrevTrack / 255)
        #disp = frame + track
        disp = frame

        disp = disp * 255
        disp = np.clip(disp, 0, 255)
        disp = disp.astype(np.uint8)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)

        #cv2.drawContours(disp, contours, -1, (0, 0, 255))
        pointMap = dict()
        centers = []
        for c in contours:
            if cv2.contourArea(c) >= 10:
                m = cv2.moments(c)
                center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
                centers.append((center[0], center[1], frameIndex))
                cv2.circle(disp, center, 3, (0, 0, 255))
      
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
                if bestTrajectories[bestCenter] == traj and mov and bestCost < 2.5 * mov + 5:
                    traj.points.append((bestCenter[0], bestCenter[1], frameIndex))
                    centers.remove(bestCenter)
                else:
                    traj.counter = traj.counter + 1

                    # Check if this trajectory is dying
                    if traj.counter >= 2:

                        # Split the trajectories into linear segments
                        splitTrajectories = traj.splitLinear()                     

                        # If we have two or more linear segments...
                        if len(splitTrajectories) >= 2:

                            # Test all possible combinations of linear segments
                            # (This is fine since there will only be a handfull of segments)
                            for linearTraj1 in splitTrajectories:
                                for linearTraj2 in splitTrajectories:
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
                                    print "TIMEDIST SAMETRAJ: ", timeDistance
                                    print "SPACEDIST SAMETRAJ: ", physicalDistance
                                    if timeDistance >= -1 and timeDistance <= 3 and physicalDistance < 35:
                                        # Check for lined up trajectories
                                        v1 = (linearTraj1.points[-1][0] - linearTraj1.points[0][0],
                                              linearTraj1.points[-1][1] - linearTraj1.points[0][1])
                                        v2 = (linearTraj2.points[-1][0] - linearTraj2.points[0][0],
                                              linearTraj2.points[-1][1] - linearTraj2.points[0][1])
                                        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                                        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

                                        cosTheta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)

                                        if linearTraj1 in splitTrajectories:
                                            splitTrajectories.remove(linearTraj1)

                                        print "COSTHETA: ", cosTheta
                                        if cosTheta > 0.9:
                                            print "COMBINING SAMETRAJ"
                                            # Combine the two segments
                                            linearTraj2.points.extend(linearTraj1.points)
                                        elif cosTheta < 0.8:
                                            print "MATCH SAMERAJ"

                                            #if cosTheta > -0.97:
                                            m1, b1, r1, p1, e1 = scipy.stats.linregress([p[:2] for p in linearTraj1.points])
                                            m2, b2, r2, p2, e2 = scipy.stats.linregress([p[:2] for p in linearTraj2.points])
                                            x = (b1 - b2) / (m2 - m1)
                                            y = m1 * x + b1

                                            x = int(round(x))
                                            y = int(round(y))
                                            #else:
                                            #    x = end[0]
                                            #    y = end[1]

                                            print "INT X: ", x
                                            print "INT Y: ", y

                                            powderPoints.append((x, y, (end[2] + start[2])/2))
                                            cv2.circle(disp, (x, y), 10, (255, 0, 255))
                                            splitTrajectories.remove(linearTraj2)

                        for linearTraj1 in splitTrajectories:
                            for linearTraj2 in unmatchedSegments:
                                # Get the start point of this trajectory
                                start = linearTraj1.points[0]
                                # Get the end points of that trajectory
                                end = linearTraj2.points[-1]

                                # Calculate time & physical distances
                                timeDistance = start[2] - end[2]
                                physicalDistance = euclideanDistance(end[:2], start[:2])

                                # If matching in time and space...
                                print "TIMEDIST: ", timeDistance
                                print "SPACEDIST: ", physicalDistance
                                if timeDistance >= -1 and timeDistance <= 3 and physicalDistance < 35:
                                    # Check for lined up trajectories
                                    v1 = (linearTraj1.points[-1][0] - linearTraj1.points[0][0],
                                          linearTraj1.points[-1][1] - linearTraj1.points[0][1])
                                    v2 = (linearTraj2.points[-1][0] - linearTraj2.points[0][0],
                                          linearTraj2.points[-1][1] - linearTraj2.points[0][1])
                                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                                    cosTheta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)

                                    if linearTraj1 in splitTrajectories:
                                        splitTrajectories.remove(linearTraj1)

                                    print "COSTHETA: ", cosTheta
                                    if cosTheta > 0.9:
                                        print "COMBINING"
                                        # Combine the two segments
                                        linearTraj2.points.extend(linearTraj1.points)
                                    elif cosTheta < 0.8:
                                        print "MATCH"
                                            
                                        #if cosTheta > -0.97:
                                        m1, b1, r1, p1, e1 = scipy.stats.linregress([p[:2] for p in linearTraj1.points])
                                        m2, b2, r2, p2, e2 = scipy.stats.linregress([p[:2] for p in linearTraj2.points])

                                        print "M1: ", m1
                                        print "M2: ", m2
                                        print "B1: ", b1
                                        print "B2: ", b2

                                        x = (b2 - b1) / (m1 - m2)
                                        y = m1 * x + b1

                                        x = int(round(x))
                                        y = int(round(y))
                                        #else:
                                        #    x = end[0]
                                        #    y = end[1]

                                        print "INT X: ", x
                                        print "INT Y: ", y
                                        #for p in linearTraj1.points:
                                        #    cv2.circle(disp, p[:2], 5, (255, 0, 0))
                                        #for p in linearTraj2.points:
                                        #    cv2.circle(disp, p[:2], 5, (0, 0, 255))
                                        #cv2.putText(disp, str(linearTraj1.points[0][2]), linearTraj1.points[0][:2], cv2.FONT_HERSHEY_PLAIN, 0.86, (0, 0, 0))
                                        #cv2.putText(disp, str(linearTraj2.points[-1][2]), linearTraj2.points[-1][:2], cv2.FONT_HERSHEY_PLAIN, 0.86, (0, 0, 0))
                                        cv2.circle(disp, (x, y), 10, (255, 0, 255))
                                        powderPoints.append((x, y, (end[2] + start[2])/2))
                                        unmatchedSegments.remove(linearTraj2)

                        #for splitTraj in splitTrajectories:
                        #    for p in splitTraj.points:
                        #        cv2.circle(disp, p[:2], 3, (255, 255, 255))
                        unmatchedSegments.extend(splitTrajectories)

                        # Finally, kill the trajectory
                        trajectories.remove(traj)

        bestCenters =dict()
        for center in centers:
            if prevCenters:
                bestDistance = ((center[0] - prevCenters[0][0])**2 + (center[1] - prevCenters[0][1])**2)**0.5
                bestCenter = prevCenters[0]
                for prevCenter in prevCenters:
                    cv2.circle(disp, prevCenter[:2], 3, (255, 0, 0))
                    distance = ((center[0] - prevCenter[0])**2 + (center[1] - prevCenter[1])**2)**0.5
                    if distance < bestDistance:
                        bestDistance = distance
                        bestCenter = prevCenter
                bestCenters[center] = bestCenter
        #for center in bestCenters:
        #    cv2.line(disp, center, bestCenters[center], (0, 255, 255), 2)

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
        #for center in prevBestCenters:
        #    cv2.line(disp, center, prevBestCenters[center], (255, 255, 0), 1)


        matches = dict()
        for center in bestCenters:
            prevCenter = bestCenters[center]
            if prevBestCenters[prevCenter] == center:
                matches[center] = prevCenter

        for center in matches:
            prevCenter = matches[center]
            #cv2.line(disp, center, matches[center], (255, 0, 255))
            trajectories.append(trajectory([prevCenter, center]))
            centers.remove(center)

        for traj in trajectories:
            #if len(traj.points) < 5:
            #    continue
            prevCenter = traj.points[0][:2]
            for center in traj.points[1:]:
                cv2.line(disp, prevCenter[:2], center[:2], (255, 0, 255), 1)
                prevCenter = center

        out.write(disp)
        cv2.imshow('frame', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prevCenters = centers
        prevFrame = frame
        prevPrevPrevTrack = prevPrevTrack
        prevPrevTrack = prevTrack
        prevTrack = track
        
    else:
        break

    frameIndex = frameIndex + 1

frameIndex = 0
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():
    ret, frame = grabframe(cap)

    if ret == True:
        for p in powderPoints:
            if p[2] >= frameIndex - 3 and p[2] <= frameIndex + 3:
                cv2.circle(frame, p[:2], 7, (0, 0, 255))
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    frameIndex = frameIndex + 1

out.release()
cap.release()

cv2.destroyAllWindows()
