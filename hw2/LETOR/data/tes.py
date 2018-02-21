import numpy as np
import collections
import random
from scipy.cluster.vq import vq, whiten, kmeans2


def getCenters(M, Training):
    centers = kmeans2(Training,M,10,minit='points')
    return centers


Training = np.load("Validation.npy")
Training = Training[: , 0:len(Training[0])-1] # drop the last column as its output label
np.savetxt('Training.csv',Training)
print len(Training)
M = 3
centers = getCenters(M, Training)
labels = centers[1]
labelModifier = [[0 for x in range(1)] for y in range(len(labels))]

for i in range(len(labels)):
    labelModifier[i][0] = labels[i]

print labelModifier
spreadList = []
for i in range(M):
    spreadList.append(np.identity(46)*0.5)
spread = np.array(spreadList)
print spread











