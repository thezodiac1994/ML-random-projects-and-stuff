import numpy as np
import os

def predict(W, X):
    #print(len(W) , len(X));
    ans = 0
    for i in range(0, len(W)):
        ans = ans + W[i] * X[i]
    return ans


directory = os.path.dirname(__file__)

W_sgd = np.load('W_cfs.npy');
path = os.path.join(directory[0:-3],'data/Validation.npy')  # we have dropped last 3 chars of directory to drop the sgd folder and move to data folder instead

intital_validation = np.load(path)
outputs = intital_validation[:,-1]

path = os.path.join(directory[0:-3],'data/Phi_Validation.npy')  # we have dropped last 3 chars of directory to drop the sgd folder and move to data folder instead
Validation = np.load(path)


RMSE = 0
SSE = 0

for i in range(0, len(Validation)-1):
    predicted = predict(W_sgd,Validation[i,:])
    actual = outputs[i]
    print(actual, predicted)
    error = actual - predicted
    SSE += (error**2)

RMSE = (2*SSE /(len(Validation)))**0.5

print (RMSE)
myfile = open("result.txt", "a")
myfile.write("RMSE = " +  str(RMSE) + "\n");