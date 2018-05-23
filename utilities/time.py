from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.random
import random 
import matplotlib.pyplot as plt
import csv
import sys
import math
import random

def getCentroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 1]) 
    sum_y = np.sum(arr[:, 2])
    return np.array([sum_x/length, sum_y/length])

RADIUS = 1

if len(sys.argv) != 2:
    print "USAGE: ", sys.argv[0] + " filename"
    exit(-1)

inputFileName = sys.argv[1]

rawFile = open(inputFileName, 'rb')
csvReader = csv.reader(rawFile, delimiter = ',')

points = np.array([np.array([float(item) for item in row]) for row in csvReader])

bestSeed = points[0]
bestCount = sum([1 for p in points if np.linalg.norm(p[1:] - points[0][1:]) < RADIUS])
bestInliers = np.array([p for p in points[1:] if np.linalg.norm(p[1:] - points[0][1:]) < RADIUS])
for x in xrange(0, 50):
    seed = random.choice(points[1:])[1:]
    inliers = np.array([p for p in points[1:] if np.linalg.norm(p[1:] - seed) < RADIUS])
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
    inliers = np.array([p for p in points[1:] if np.linalg.norm(p[1:] - centroid) < radius])
    radius = radius * 1.02

print "numInliers: ", len(inliers)
print "radius: ", radius

print "total particle count: ", len(points)

filtered = np.array([p for p in points if np.linalg.norm(p[1:] - centroid) < 2])[:,0]
filtered = np.sort(filtered)
print "max(filtered): ", max(filtered)
print "len(filtered): ", len(filtered)
print "material efficiency: %", 100.0 * float(len(filtered)) / len(points)

#mu = 1.8e-6
#sigma = 3.3e-7
#masses = np.random.normal(mu, sigma, len(filtered))
#print "Total mass: ", sum(masses)

window = int((4.0 / (600.0 / 60)) * 4500)
print "window: ", window
sim = []
ts = []
for f in filtered:
	if f >= window:
		ts.append(f)
		sim.append(sum([1 for _f in filtered if _f < f and _f > f - window]))

#startvolume = 300
#sim.append(startvolume)
#ts.append(0)
#newsum = startvolume
#
#rate = float(len(filtered))/max(filtered)
#for i in xrange(1, len(filtered)-1):
#	newsum = newsum - (float(filtered[i]) - filtered[i-1])*rate
#	sim.append(newsum)
#	ts.append(filtered[i])
#	newsum = newsum + 1
#	sim.append(newsum)
#	ts.append(filtered[i])

print "mean: ", np.mean(sim)
print "standard deviation: ", np.std(sim)
print "standard deviation: %", 100*np.std(sim)/np.mean(sim)

ax = plt.subplot(111)
ax.plot(ts, sim)
ax.set_xlim([window, max(filtered)])
ax.set_ylim([0, max(sim)*1.1])
plt.show()

with open('time.csv', 'w') as f:
	for t, s in zip(ts, sim):
		f.write(str(t) + ',' + str(s) + '\n')

