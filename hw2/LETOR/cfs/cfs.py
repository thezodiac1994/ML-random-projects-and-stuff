import numpy as np
import random
import os


def cfs(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(
    L2_lambda * np.identity(design_matrix.shape[1]) +
    np.matmul(design_matrix.T, design_matrix),
    np.matmul(design_matrix.T, output_data)
    ).flatten()


directory = os.path.dirname(__file__)
path = os.path.join(directory[0:-3],'data/Phi_Training.npy')  # we have dropped last 3 chars of directory to drop the sgd folder and move to data folder instead
Training = np.load(path)

path = os.path.join(directory[0:-3],'data/Training.npy')
initial_set = np.load(path)

outputs = initial_set[:,-1]
#print(Training[1:10,:]);
lamda = 0.1
W = cfs(lamda,Training,outputs)
np.save("W_cfs.npy",W)
np.savetxt("W_cfs.txt",W)

myfile = open("result.txt", "a")
myfile.write("M = " + str(len(Training[0])) +  " lambda = " + str(lamda) + " ");