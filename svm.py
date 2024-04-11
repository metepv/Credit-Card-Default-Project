"""
SUPPORT VECTOR MACHINE
WORK IN PROGRESS.............
"""
import numpy as np
import random
import math

def data_to_matrix_function():
    #this function will take the raw data and column size
    #and will return the main X decision matrix as an output.

    raw_data = open("UCI_Credit_Card.csv",'r')
    #the raw data obtained from kaggle.com as below:
    #https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download

    column_size = int(input("Specified column size to be imported: "))
    #column size will be input for model training with X which will have variying column size
    #there are 23 number of fixed features.
    #X1,X2,X3,X4,...,X23.
    #decision_matrix_function(raw_data, column_size)
    
    predictor_matrix = [] 
    response_vector = []
     
    for i in range(1,column_size+2):                        #first two line are predictor names.
        
        column_vector_str = raw_data.readline()
        """
        try:
            first = column_vector_str[0] #if a line starts with a number it will
            first = int(first)           #it will be added as column to the matrix.
            
            column_vector_lst = column_vector_str.split(',') #splitting the line
                                                             # to add it to a column vector.
            
            column_vector_lst[-1] = column_vector_lst[0:-2]  #last element is \n
                                                             # which is removed.
            column_vector_int_lst = []                                                 
            for i2 in range(0,len(column_vector_lst)-1):
                 column_vector_int_lst.append(int(column_vector_lst[i2]))
                
            X.append(column_vector_int_lst)                      
            
        except ValueError:
            continue
        """

        #due to the integer-float-string conversion the code above generates
        #matrix with column dimension less than the input value.

        if column_vector_str[0].isdigit():
            column_vector_lst = column_vector_str.split(',')
            column_vector_lst[-1] = column_vector_lst[-1][0:1]
            response_vector.append(int(column_vector_lst[-1]))

            column_vector_float_lst = [1]                                                 
            for i2 in range(1,len(column_vector_lst)-1): # last value is the y value of the regression.
                 column_vector_float_lst.append(float(column_vector_lst[i2]))

            predictor_matrix.append(column_vector_float_lst)
            
        else:
            continue
    
    
    #NORMALIZATION
    response_vector=(response_vector-(np.mean(response_vector)))/np.std(response_vector)
    predictor_matrix=(predictor_matrix-(np.mean(predictor_matrix)))/np.std(predictor_matrix)
    #print(np.mean(response_vector))
    # response_vector=random.shuffle(response_vector)
    # predictor_matrix=random.shuffle(predictor_matrix)
    trainset_length=math.floor(len(predictor_matrix)*0.7)
    
    X_test=predictor_matrix[trainset_length:]
    X_train=predictor_matrix[0:trainset_length]
    Y_test=response_vector[trainset_length:]
    Y_train=response_vector[:trainset_length]
    return predictor_matrix, response_vector, X_test, X_train, Y_test, Y_train

data_to_matrix_function()

def gradient_descent(X,Y,C,learning_rate,tolerance):

    gradient=0
    for k in range(0,len(X)):

        gradient+=C*Y[k]*X[k]

    W=np.zeros(len(X[0]))

    while(gradient*learning_rate>tolerance):
        W=W-learning_rate*gradient
    
    return W


def Support_Vector_Machine(X_Train,X_Test,Y_Train,Y_Test,iteration,tolerance,learning_rate,C):

    for k in Y_Train:
        if k==0:
            Y_Train[Y_Train.index(k)]=-1

    W=gradient_descent(X_Train,Y_Train,C,learning_rate,tolerance)

    training_results=[]
    test_results=[]

    for column in X_Train:
        if np.dot(W,column)>0:
            training_results.append(1)
        else:
            training_results.append(0)

    for X in X_Test:
        if np.dot(W,X)>0:
            test_results.append(1)
        else:
            test_results.append(0)

    return training_results,test_results
