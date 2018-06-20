"""
Tracking algorithm major sub-algorithms. Bulk of the tracking computation 
perfomed here.

Connor Coward
19 June 2018
"""

import cv2
import numpy as np
import scipy.stats
import util

def getCenters(diff):
	"""Blob detection. Finds potential particle locations in a frame by first 
       performing a binary threshold operation, dilation, erosion, and then blob
       detection"""
    ret, track = cv2.threshold(diff, 128, 256, cv2.THRESH_BINARY)
    kernelOpen = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, kernelOpen)
    track = cv2.morphologyEx(track, cv2.MORPH_OPEN, kernelOpen)

    _, contours, _ = cv2.findContours(track,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        if cv2.contourArea(c) >= 1:
            m = cv2.moments(c)
            center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
            centers.append((center[0], center[1]))

    return centers, track


def getParticlePoints(cap, medianImage):
	"""Loops through entire video, extracting all potential particle locations 
       in the video"""

    particlePoints = dict()

    # Go back to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frameindex = 0
    while cap.isOpened():
        if frameindex % 10 == 0:
            print "FRAME: ",  frameindex

        ret, frame = util.grabframe(cap)

        if ret is True:
            diff = np.subtract(frame, medianImage)

            scaled = util.scaleOneSided(diff)

            centers, blobImage = getCenters(scaled)

            if centers:
                particlePoints[frameindex] = centers

            cv2.imshow('Blobs', blobImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # Keep track of a frame index
        frameindex = frameindex + 1

    return particlePoints


def getSegmentsFromPoints(particlePoints, shape, maxT):
	"""Attempts to form linear trajectories from points in the frame"""
	segments = []
	segmentIdMap = dict()
	segmentIdCounter = 0
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
			while t2 not in particlePoints and t2 < maxT:
				t2 = t2 + 1

			if t2 >= maxT:
				continue

			p2, _ = util.findClosest(p1, particlePoints[t2])

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
					   t < maxT and skipCounter < 7):
					if t in particlePoints:
						cv2.circle(debugImage, (int(x), int(y)), 4, (255, 0, 0))
						closest, distance = util.findClosest(
												(x, y), 
												particlePoints[t])
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
						closest, distance = util.findClosest(
												(x, y), 
												particlePoints[t])
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

				mX, bX, _, _, _ = scipy.stats.linregress(
				    [(inlier[0], inlier[1]) for inlier in inliers])
				mY, bY, _, _, _ = scipy.stats.linregress(
				    [(inlier[0], inlier[2]) for inlier in inliers])

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
					   t < maxT and skipCounter < 3):
					if t in touchedPoints:
						closest, distance = util.findClosest(
                            (x, y), touchedPoints[t])
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
						closest, distance = util.findClosest(
							(x, y), touchedPoints[t])
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
				segmentIdMap[segments[-1]] = segmentIdCounter
				segmentIdCounter = segmentIdCounter + 1

	return segments, segmentIdMap


def getCollisionsFromSegments(segments, segmentIdMap):
	"""Matches pairs of segments which may together form the trajectory of a 
       single particle that collides with a planar substrate. Finds situations 
       where one segment ends just as the next is beginning (which indicates 
       that the two segments are actually from the same particle after it has 
	   bounced"""
	collisions = []
	for i in xrange(0, len(segments)):
		segment1 = segments[i]

		bestDistance = float('inf')
		bestMatch = None
		bestSegment = None

		searchFrameMin = max(i - 300, 0)
		searchFrameMax = min(i + 300, len(segments))
		for segment2 in segments[searchFrameMin:searchFrameMax]:
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

			distance = util.euclideanDistance((xBeg2, yBeg2), (xEnd1, yEnd1))

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
			print ("Matched Ends!", bestMatch, 
				"Segments:", segmentIdMap[segment1],
				segmentIdMap[bestSegment])

	return collisions

