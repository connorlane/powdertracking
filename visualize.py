
import cv2
import pickle

def playbackResults(cap, collisions, segments, segmentIdMap, scale):
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	newWidth = int(width * scale)
	newHeight = int(height * scale)

	# Create a video writer for visualizing the results
	linesVid = cv2.VideoWriter('lines.avi',
							   cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
						       50,
						       (newWidth, newHeight))

	# Go back to first frame
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	frameindex = 0
	while cap.isOpened():
		ret, frame = cap.read()

		if ret is True:
			frame = cv2.resize(frame, (newWidth, newHeight))
			for segment in segments:
				if (frameindex > segment[4]) and (frameindex < segment[5]):
					t1 = segment[4]
					x1 = int(scale * (segment[0] * segment[4] + segment[1]))
					y1 = int(scale * (segment[2] * segment[4] + segment[3]))
					x2 = int(scale * (segment[0] * frameindex + segment[1]))
					y2 = int(scale * (segment[2] * frameindex + segment[3]))
					cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
					cv2.putText(frame,
								str(segmentIdMap[segment]),
								(x2 + 2, y2 + 2),
								cv2.FONT_HERSHEY_SIMPLEX,
								0.4,
								(255, 255, 255))

			for t, x, y in collisions:
				if t > frameindex - 5 and t < frameindex + 5:
					x_scaled = int(scale * x)
					y_scaled = int(scale * y)
					cv2.circle(frame, (x_scaled, y_scaled), 4, (0, 0, 255))

			cv2.imshow('segments', frame)
			linesVid.write(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

		# Keep track of a frame index
		frameindex = frameindex + 1

	linesVid.release()

	return

