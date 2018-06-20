"""
A basic video playback script. Allows navigation forward and backward in 
the video.

Connor Coward
19 June 2018
"""

import cv2
import sys

FRAMEWIDTH = 600

if len(sys.argv) != 2:
    print 'USAGE:', sys.argv[0], '<video>'
    exit(-1)

cap = cv2.VideoCapture(sys.argv[1], cv2.WINDOW_NORMAL)

ret, frame = cap.read()
height, width = frame.shape[:2]
print "Frame width:", width
print "Frame height:", height

ratio = FRAMEWIDTH / float(width)
newFrameWidth = int(width * ratio)
newFrameHeight = int(height * ratio)

while(True):
    frameResized = cv2.resize(frame, (newFrameWidth, newFrameHeight))
    cv2.imshow('Press \'q\' to quit, any key to advance frame', frameResized)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
