import numpy as np

Matrix = np.load("Matrix.npy")
Training,Validation,Testing = np.split(Matrix, [int(0.8*len(Matrix)) , int(0.9*len(Matrix))]) # split from 0 to 0.8 0.8 to 0.9 and 0.9 ro 1.0
#print(len(Matrix))
#print(len(Training))
#print(len(Testing))
#print(len(Validation))

np.save("Training.npy",Training)
np.save("Testing.npy",Testing)
np.save("Validation.npy",Validation)


