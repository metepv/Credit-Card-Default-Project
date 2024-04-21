"""
EEE 485 PROJECT: Credit Card Default

Module 2:This module takes the decision matrix X and the response vector Y to make prediction
         with logistic regression.
"""
import numpy as np
import LR_Pre as LR_Pre

def gradient_ascent():

    (predictor_matrix_numpy, response_vector, response_vector_org) = LR_Pre.data_to_matrix_function()
    number_of_iteration = int(input("Iteration number: "))
    X = np.array(predictor_matrix_numpy)        # X = [1,X1,X2,X3,...,X23] the decision matrix
    Y = np.array(response_vector)               # Y is the response vector
    Y_un = np.array(response_vector_org)

    #print(X)
    #print(Y)
    #print(Y_un)

    Beta_coefficients = np.ones(24)*0.5  # B = [B0,B1,B2,B3,...B23], unknown random coefficients which
                                         # will be updated in gradient ascent algorithm.

    Beta_new = np.zeros(24)
    Beta_old = Beta_coefficients

    row_size = X.shape[0]

    #Pi vector generation:
    Pi_vector = []                          
    for i2 in range(0, row_size):
        Pi_vector.append(logistic_function(Beta_old,X[i2,:]))

    #W matrix generation:
    W = np.eye(X.shape[0])
    for i in range(0, X.shape[0]):
        W[i,i] = logistic_function(Beta_old,X[i,:])*(1-logistic_function(Beta_old,X[i,:]))

    X_T = np.transpose(X)
    A = np.dot(np.dot(X_T,W), X)  #A = X_t*W*X
    A_inverse = np.linalg.inv(A)  #A^-1 = (X_t*W*X)^-1

    
    for i in range(0,number_of_iteration):

        A = np.dot(np.dot(X_T,W), X) #A = X_t*W*X
        A_inverse = np.linalg.inv(A) #A^-1 = (X_t*W*X)^-1
        C1 = np.dot(A_inverse,X_T)
        C2 = Y - Pi_vector
        C3 = np.dot(C1,C2)
        Beta_new = Beta_old + C3

        #updating matrices and coefficients.
        Beta_old = Beta_new
        
        for i in range(0, X.shape[0]):
            Pi_vector[i] = (logistic_function(Beta_old,X[i,:])) 

        for i in range(0, X.shape[0]):
            W[i,i] = logistic_function(Beta_old,X[i,:])*(1-logistic_function(Beta_old,X[i,:]))

    return Beta_new, X, Y, Y_un
    

def logistic_function(beta_vector,X_i_colum_vector):

    logistic_result = np.longdouble()
    dot_product = np.longdouble()
    dot_product = np.dot(beta_vector,X_i_colum_vector)/10000

    logistic_result = np.exp(dot_product)/(1 + np.exp(dot_product))
    
    return logistic_result



