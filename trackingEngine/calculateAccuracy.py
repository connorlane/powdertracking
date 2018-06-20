"""Simple helper utility to calculate tracking accuracy by comparing manual tracking results and software-generated resutls

Connor Coward
19 June 2018
"""

import csv
import sys
import numpy as np
import os

if len(sys.argv) != 3:
    print "USAGE: ", sys.argv[0] + " <video filename> <num frames to search>"
    exit(-1)

inputFileName = sys.argv[1]
numFrames = int(sys.argv[2])

fileNameOnly = os.path.basename(inputFileName)
baseName = os.path.splitext(fileNameOnly)[0]

fileNames = [baseName + '.csv', baseName + '_falsePositives.csv', baseName + '_falseNegatives.csv']
files = [open(fn, 'rb') for fn in fileNames]
readers = [csv.reader(f, delimiter = ',') for f in files]
pointLists = [np.array([np.array([float(item) for item in row]) for row in r]) for r in readers]
counts = [sum(p[0] < numFrames for p in points) for points in pointLists]

calculatedResultsCount = counts[0]
falsePositivesCount = counts[1]
falseNegativesCount = counts[2]

falsePositivesPercent = float(falsePositivesCount) / (calculatedResultsCount + falseNegativesCount)
falseNegativesPercent = float(falseNegativesCount) / (calculatedResultsCount + falseNegativesCount)
accuracy = float(calculatedResultsCount - falsePositivesCount) / (calculatedResultsCount + falseNegativesCount)

print "False Positives Count: {0}".format(falsePositivesCount)
print "False Negatives Count: {0}".format(falseNegativesCount)
print "Accuracy Count: {0}".format(calculatedResultsCount)
print "False Positives: {0:.2%}".format(falsePositivesPercent)
print "False Negatives: {0:.2%}".format(falseNegativesPercent)
print "Accuracy: {0:.2%}".format(accuracy)
