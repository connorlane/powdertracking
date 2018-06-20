"""Generates a heatmap image and surface plot from a list of particle collision 
   locations generated by the tracking software.

Connor Coward
19 June 2018
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import csv
import sys
import random

def getCentroid(arr):
	"""Calculate the centroid of a 2d array where each row is a 2d point"""
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.array([sum_x/length, sum_y/length])


RADIUS = 0.01 # Inlier search radius used to find the center of the data
WIDTH = 0.5 # Physical width range to display on the heat map

# Basic input sanity check
if len(sys.argv) != 2:
    print "USAGE: ", sys.argv[0] + " filename"
    exit(-1)

# Grab the input csv file name provided
inputFileName = sys.argv[1]

# Create a csv reader object for parsing the csv file
rawFile = open(inputFileName, 'rb')
csvReader = csv.reader(rawFile, delimiter = ',')

# Pull all the data in the csv file into a list
points = np.array([np.array(
	[float(item) for item in row[1:3]]) for row in csvReader])

# Perform a simple RANSAC to find the center of the points with 
#     possible extreme outliers
bestSeed = points[0]
bestCount = sum([1 for p in points if np.linalg.norm(p - points[0]) < RADIUS])
bestInliers = np.array(
	[p for p in points[1:] if np.linalg.norm(p - points[0]) < RADIUS])

for x in xrange(0, 100):
    seed = random.choice(points[1:])
    inliers = np.array(
		[p for p in points[1:] if np.linalg.norm(p - seed) < RADIUS])
    if len(inliers) > bestCount:
        bestCount = len(inliers)
        bestSeed = seed
        bestInliers = inliers
print "bestSeed:", bestSeed
print "bestCount:", bestCount

radius = RADIUS
inliers = bestInliers
while len(inliers) < 0.5 * len(points):
    centroid = getCentroid(inliers)
    inliers = np.array(
		[p for p in points[1:] if np.linalg.norm(p - centroid) < radius])
    radius = radius * 1.02

print "numInliers:", len(inliers)

# Offset each point by the centroid
x = [px - centroid[0] for px in points[:,0]]
y = [py - centroid[1] for py in points[:,1]]

# Calculate the heat map values
heatmap, xedges, yedges = np.histogram2d(
	x, y, bins=15, range=((-WIDTH, WIDTH),(-WIDTH, WIDTH)))

# Normalize
heatmap = heatmap * 1 / np.max(heatmap)

# Convert to grip for matplotlib
xx, yy = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]

# Scale & convert to inches
xx = (xx * 0.25 / heatmap.shape[0] - 0.125) * 25.4
yy = (yy * 0.25 / heatmap.shape[1] - 0.125) * 25.4

# Create surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Powder Particle Spatial Distribution')
ax.set_xlabel('(mm)')
ax.set_ylabel('(mm)')
ax.set_zlabel('Relative Particle Frequency')
surf = ax.plot_surface(
	xx, yy, heatmap, rstride = 1, cstride = 1, cmap='plasma', 
	linewidth=0, antialiased=False)

# Display the surface map
plt.show()

# Extent for heatmap
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Show heat map
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='plasma')
plt.show()

