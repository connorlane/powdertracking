#! /usr/bin/env python

import scipy.stats
import math

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
    return ((p1[0] - p2[0])**2 + (p1[1] -p2[1])**2)**0.5

def projectPoint(m, b, point):
    x = point[0]
    y = point[1]
    unit_vector = 1 / math.sqrt(m*m + 1), m / math.sqrt(m*m + 1)
    print "UNIT_VECT: ", unit_vector
    scalar_projection = (x + m * (y - b)) / math.sqrt(m*m + 1)
    print "SCALAR_PROJ: ", scalar_projection
    proj_x = int(round(scalar_projection * unit_vector[0]))
    proj_y = int(round(scalar_projection * unit_vector[1] + b))
    return proj_x, proj_y

class trajectory:
    def __init__(self, points = []):
        self.points = points
        self.counter = 0

    def fitLine(self):
        print "POINTS: ", self.points[:-1]
        slope, intercept, r, p, stderr = scipy.stats.linregress(self.points)
        return slope, intercept

    def movement(self):
        return ((self.points[-1][0] - self.points[-2][0])**2 + (self.points[-1][1] - self.points[-2][1])**2)**0.5

    def distance(self):
        p1 = self.points[0]
        distance = 0
        for p2 in self.points[1:]:
            distance = distance + euclideanDistance(p1, p2)
        return distance

    def velocity(self):
        return self.distance() / (len(self.points) - 1)

    def getCost(self, c):
        m, b = self.fitLine()
        d = self.velocity()

        print "D: ", d
        print "M: ", m
        print "B: ", b

        dX = self.points[-1][0] - self.points[-2][0]
        dY = self.points[-1][1] - self.points[-2][1]

        if not math.isnan(m):
            proj1 = projectPoint(m, b, self.points[1])
            proj2 = projectPoint(m, b, self.points[-1])

            dX = (proj1[0] - proj2[0]) / (len(self.points) - 1)
            dY = (proj1[1] - proj2[1]) / (len(self.points) - 1)

            #cv2.circle(disp, proj1, 4, (0, 255, 0))
            #cv2.circle(disp, proj2, 4, (0, 0, 255))

        targetX = int(round(self.points[-1][0] - dX * (self.counter + 1)))
        targetY = int(round(self.points[-1][1] - dY * (self.counter + 1)))
        
        #cv2.circle(disp, (targetX, targetY), 4, (255, 0, 0))

        cost = ((c[0] - targetX)**2 + (c[1] - targetY)**2)**0.5

        return cost

cap = cv2.VideoCapture('../Video/ti.mov') 
ret, prevFrame = grabframe(cap)
prevTracks = [np.zeros(prevFrame.shape[:2])] * 10
prevTrack = np.zeros(prevFrame.shape[:2])
prevPrevTrack = np.zeros(prevFrame.shape[:2])
prevPrevPrevTrack = np.zeros(prevFrame.shape[:2])

cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

print prevFrame.shape

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
frameIndex = 0

while cap.isOpened():
    ret, frame = grabframe(cap)

    if ret == True:
        diff = np.subtract(frame, prevFrame)

        Gx = cv2.Sobel(diff, -1, 1, 0)
        Gy = cv2.Sobel(diff, -1, 0, 1)

        Gx2 = np.square(Gx)
        Gy2 = np.square(Gy)

        Gmag = np.sqrt(Gx2 + Gy2)

        diff = scaleOneSided(Gmag)

        ret, track = cv2.threshold(diff, 128, 256, cv2.THRESH_BINARY)
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
                centers.append(center)
                #cv2.circle(disp, center, 3, (0, 0, 255))
      
        bestTrajectories = dict()
        for c in centers:
            if trajectories:
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
                bestCost = traj.getCost(centers[0])
                bestCenter = centers[0]
                for c in centers[1:]:
                    cost = traj.getCost(c)
                    if cost < bestCost:
                        bestCenter = c
                        bestCost = cost
                print "BEST COST: ", bestCost
                mov = traj.movement()
                print "MOVEMENT: ", mov
                if bestTrajectories[bestCenter] == traj and mov and bestCost < 4*mov:
                    cv2.putText(disp, str(bestCost), bestCenter, cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
                    traj.points.append(bestCenter)
                    centers.remove(bestCenter)
                else:
                    traj.counter = traj.counter + 1
                    if traj.counter > 2:
                        trajectories.remove(traj)


        print "TRAJECTORIES"
        print trajectories


        bestCenters = dict()
        for center in centers:
            if prevCenters:
                bestDistance = ((center[0] - prevCenters[0][0])**2 + (center[1] - prevCenters[0][1])**2)**0.5
                bestCenter = prevCenters[0]
                for prevCenter in prevCenters:
                    cv2.circle(disp, prevCenter, 2, (0, 0, 0))
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

        print "MATCHES"
        print matches
        for center in matches:
            prevCenter = matches[center]
            cv2.line(disp, center, matches[center], (255, 0, 255))
            trajectories.append(trajectory([prevCenter, center]))
            centers.remove(center)

        print "MISFITS"
        print centers

        for traj in trajectories:
            prevCenter = traj.points[0]
            for center in traj.points[1:]:
                slope, intercept = traj.fitLine()
                if slope > 0:
                    color = (255, 0, 255)
                else:
                    color = (255, 255, 0)
                if len(traj.points) > 5:
                    cv2.line(disp, prevCenter, center, color, 1)
                prevCenter = center

        out.write(disp)
        cv2.imshow('frame', disp)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        prevCenters = centers
        prevFrame = frame
        prevPrevPrevTrack = prevPrevTrack
        prevPrevTrack = prevTrack
        prevTrack = track
        
    else:
        break

    frameIndex = frameIndex + 1

out.release()
cap.release()
cv2.destroyAllWindows()

