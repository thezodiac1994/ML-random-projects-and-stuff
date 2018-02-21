import numpy as np
import os

directory = os.path.dirname(__file__)
path = os.path.join(directory,"MQ2007/Querylevelnorm.txt")

F = open(path, 'r')
counter = 0
# Matrix will have 0 45 : features and 46 : label

while(1):
    current_line = F.readline()  # A is used for reading elements from F file
    if(current_line == ''):
        break

    words = current_line.split()
    current_row = [None] * 47
    current_row[46] = int(words[0])  # this is the output label

    words = words[2:48] # these are the 46 features

    for i in range(0,46) :
        words[i] = words[i].split(':')  # split with respect to ':'
        current_row[i] = float(words[i][1]) # after splitting wrt ':' , get the part to its right

    if (counter == 0):
        Matrix = np.array(current_row)
        counter = counter+1
    else :
        Matrix = np.vstack ( [Matrix, current_row] )
        counter = counter + 1


np.random.shuffle(Matrix) # random shuffling
# drop rows with all 0 or nan values
mask = np.all(np.isnan(Matrix) | np.equal(Matrix, 0), axis=1)
Matrix = Matrix[~mask] # 3 rows were dropped


np.save("Matrix.npy",Matrix)
np.savetxt('Matrix.csv',Matrix)


#z = np.load("Matrix.npy")
#print(z)

