import numpy as np
import scipy.cluster as spc

def getCenters(M, Data):

    Centers,labels = spc.vq.kmeans2(Data,M,minit='points',missing='warn')
    Centers = Centers[:, np.newaxis, :]
    return Centers,labels

def generateSpread(M,centers,label,data):

    spread = []
    for i in range(0,M):
        cluster = np.zeros(len(data[0]));
        for j in range(0,len(data)):
            if(label[j]==i):
                cluster = np.vstack((cluster,data[j]))
        cluster = cluster[1:len(cluster)]

        sigma = np.multiply(np.cov(cluster.T),np.identity(46))*0.1 # 46 x 46
        spread.append(sigma)
        print(len(sigma),len(sigma[0]))

    return np.array(spread)
    #return np.array(spread)


def compute_design_matrix(X, centers, spreads):
    # use broadcast
    X = X[np.newaxis,:,:]
    basis_func_outputs = np.exp(
    np.sum(
    np.matmul(X - centers, spreads) * (X - centers),
    axis=2
    ) / (-2)
    ).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)



M = 40
Training = np.load("Training.npy")
Training = Training[: , 0:len(Training[0])-1] # drop the last column as its output label
Centers,labels = getCenters(M, Training)
Spread = generateSpread(M,Centers,labels,Training)
Phi_Training = compute_design_matrix(Training,Centers,Spread)
np.save('Phi_Training.npy',Phi_Training)
np.savetxt('Phi_Training.csv',Phi_Training)

Validation = np.load('Validation.npy')
Validation = Validation[:,0:len(Validation[0])-1]
Phi_Validation = compute_design_matrix(Validation,Centers,Spread)
np.save('Phi_Validation.npy',Phi_Validation)
np.savetxt('Phi_Validation.csv',Phi_Validation)

Testing = np.load('Testing.npy')
Testing = Testing[:,0:-1]
Phi_Testing = compute_design_matrix(Testing,Centers,Spread)
np.save('Phi_Testing.npy',Phi_Validation)
np.savetxt('Phi_Testing.csv',Phi_Testing)

#print(len(Phi_Training[0]) ,len(Phi_Validation[0]), len(Phi_Testing[0]))