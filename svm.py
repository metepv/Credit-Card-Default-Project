"""
SUPPORT VECTOR MACHINE
WORK IN PROGRESS.............
"""
import numpy as np

def gradient_descent(X,Y,C,learning_rate,tolerance):

    gradient=0
    for k in range(0,len(X)):

        gradient+=C*Y[k]*X[k]

    W=np.zeros(len(X))

    while(gradient*learning_rate>tolerance):
        W=W-learning_rate*gradient
    
    return W


def Support_Vector_Machine(X_Train,X_Test,Y_Train,Y_Test,iteration,tolerance,learning_rate,C):

    for k in Y_Train:
        if k==0:
            Y_Train[Y_Train.index(k)]=-1

    W=gradient_descent(X_Train,Y_Train,C,learning_rate,tolerance)

    training_results=[]

    for column in X_Train:
        if np.dot(W,column)>0:
            training_results.append(1)
        else:
            training_results.append(0)

    return training_results
