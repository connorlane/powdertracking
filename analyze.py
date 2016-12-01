#! /usr/bin/env python

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
                cv2.circle(disp, center, 3, (0, 0, 255))
      
        bestTrajectories = dict()
        for c in centers:
            if traj:
                bestCost = getCost(traj[0], c)
                bestTraj = traj[0]
                for traj in trajectories[1:]:
                    cost = getCost(traj, c)
                    if cost < bestCost:
                        bestTraj = traj
                        bestCost = cost
                bestTrajectories[c] = bestTraj

        bestCenters = dict()
        for traj in trajectories:
            if c:
                bestCost = getCost(traj, c[0])
                bestCenter = c[0]
                for c in centers[1:]:
                    cost = getCost(traj, c)
                    if cost < bestCost:
                        bestCenter = c
                        bestCost = cost
                bestCenters[traj] = bestCenter

        for traj in bestCenters:
            bestCenter = bestCenters[traj]
            if bestTrajectories[bestCenter] == traj:
                traj.append(bestCenter)
                centers.remove(bestCenter)

        print "TRAJECTORIES"
        print trajectories

        

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
