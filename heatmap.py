from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import csv
import sys
import random

def getCentroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.array([sum_x/length, sum_y/length])

RADIUS = 0.01

if len(sys.argv) != 2:
    print "USAGE: ", sys.argv[0] + " filename"
    exit(-1)

inputFileName = sys.argv[1]

rawFile = open(inputFileName, 'rb')
csvReader = csv.reader(rawFile, delimiter = ',')

points = np.array([np.array([float(item) for item in row[1:3]]) for row in csvReader])

bestSeed = points[0]
bestCount = sum([1 for p in points if np.linalg.norm(p - points[0]) < RADIUS])
bestInliers = np.array([p for p in points[1:] if np.linalg.norm(p - points[0]) < RADIUS])
for x in xrange(0, 100):
    seed = random.choice(points[1:])
    inliers = np.array([p for p in points[1:] if np.linalg.norm(p - seed) < RADIUS])
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
    inliers = np.array([p for p in points[1:] if np.linalg.norm(p - centroid) < radius])
    radius = radius * 1.02

print "numInliers:", len(inliers)

x = [px - centroid[0] for px in points[:,0]]
y = [py - centroid[1] for py in points[:,1]]

WIDTH = 0.5

heatmap, xedges, yedges = np.histogram2d(x, y, bins=15, range=((-WIDTH, WIDTH),(-WIDTH, WIDTH)))

heatmap = heatmap * 1 / np.max(heatmap)

xx, yy = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]

xx = (xx * 0.25 / heatmap.shape[0] - 0.125) * 25.4
yy = (yy * 0.25 / heatmap.shape[1] - 0.125) * 25.4

#X = []
#Y = []
#Z = []
#for row in xrange(0, heatmap.shape[0]):
#    for col in xrange(0, heatmap.shape[1]):
#        X.append(col)
#        Y.append(row)
#        Z.append(heatmap[row, col])
#
#X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title('Powder Particle Spatial Distribution')
ax.set_xlabel('(mm)')
ax.set_ylabel('(mm)')
ax.set_zlabel('Relative Particle Frequency')

surf = ax.plot_surface(xx, yy, heatmap, rstride = 1, cstride = 1, cmap='plasma', linewidth=0, antialiased=False)
#ax.set_zlim(0, 0.01)

plt.show()

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#extent = [-0.0025, 0.0025, -0.0025, 0.0025]

#plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='plasma')
plt.show()

