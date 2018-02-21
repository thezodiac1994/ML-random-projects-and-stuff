import numpy as np
import os

directory = os.path.dirname(__file__)
path = os.path.join(directory,"input.csv")
pathOutput = os.path.join(directory,"output.csv")

dataFrameInput = np.genfromtxt(path, delimiter=',')
dataFrameOutput = np.genfromtxt(pathOutput)

Matrix = np.column_stack((dataFrameInput,dataFrameOutput))
print Matrix

np.random.shuffle(Matrix) # random shuffling
# drop rows with all 0 or nan values
mask = np.all(np.isnan(Matrix) | np.equal(Matrix, 0), axis=1)
Matrix = Matrix[~mask] # 0 rows were dropped


np.save("Matrix.npy",Matrix)
np.savetxt('Matrix.csv',Matrix)


#z = np.load("Matrix.npy")
#print(z)

