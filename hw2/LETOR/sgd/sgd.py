import numpy as np
import matplotlib.pyplot as mp
import random
import os
myfile = open("result1.txt", "a")
def sgd(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, len(design_matrix[0])])
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary
    lastError = 100;
    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound]
            E_D = np.matmul(
            (np.matmul(Phi, weights.T)-t).T,
            Phi
            )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
        print (np.linalg.norm(E))
        if(np.linalg.norm(E) < lastError):
            lastError = np.linalg.norm(E)
        else:
            myfile.write("*")
            return weights[len(weights)-1]
    return weights[len(weights)-1]

directory = os.path.dirname(__file__)

path = os.path.join(directory[0:-3],'data/Phi_Training.npy')  # we have dropped last 3 chars of directory to drop the sgd folder and move to data folder instead
Phi_Training = np.load(path)

path = os.path.join(directory[0:-3],'data/Training.npy')
initial_set = np.load(path)

outputs = initial_set[:,-1]
#print(Training[1:10,:]);

design_matrix = Phi_Training
learning_rate = 0.0001
mini_batchsize = 550
num_epochs = 40
L2_lambda = 0.001
W = sgd(learning_rate,mini_batchsize,num_epochs,L2_lambda,design_matrix,outputs);
np.save("W_sgd.npy",W)
np.savetxt("W_sgd.txt",W)

myfile.write("M = " + str(len(design_matrix[0])) + " epochs = " + str(num_epochs) + " batch size = " + str(mini_batchsize)+ "  learning rate = " + str(learning_rate) + " lambda = " + str(L2_lambda));