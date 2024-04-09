"""
EEE 485 PROJECT: Credit Card Default

Module 2:This module takes the decision matrix X and the response vector Y to make prediction
         with logistic regression.
"""
import numpy as np
import Logistic_Regression_Preprocess as LRP

def gradient_ascent(X,Y,B,iteration):

    Beta_new = np.zeros(24)
    Beta_old = Beta_coefficients

    #Pi vector generation:
    Pi_vector = []                              
    for i in range(0, X.shape[0]):
        Pi_vector.append(logistic_function(Beta_old,X[i,:])) 
    #print(Pi_vector[0])

    #W matrix generation:
    W = np.eye(X.shape[0])
    for i in range(0, X.shape[0]):
        W[i,i] = logistic_function(Beta_old,X[i,:])*(1-logistic_function(Beta_old,X[i,:]))
    #print(W)

    X_T = np.transpose(X)
    A = np.dot(np.dot(X_T,W), X)  #A = X_t*W*X
    A_inverse = np.linalg.inv(A)  #A^-1 = (X_t*W*X)^-1

    
    for i in range(0,iteration):

        A = np.dot(np.dot(X_T,W), X) #A = X_t*W*X
        A_inverse = np.linalg.inv(A) #A^-1 = (X_t*W*X)^-1
 
        Beta_new = Beta_old + A_inverse * X_T * (Y - Pi_vector)

        #updating matrices and coefficients.
        Beta_old = Beta_new
        
        for i in range(0, X.shape[0]):
            Pi_vector[i] = (logistic_function(Beta_old,X[i,:])) 

        for i in range(0, X.shape[0]):
            W[i,i] = logistic_function(Beta_old,X[i,:])*(1-logistic_function(Beta_old,X[i,:]))

    return Beta_new
    


def logistic_function(beta_vector,X_i_colum_vector):
    # y = exp(XiB)/1+exp(XiB) where Xi is column vector of the decision matrix X.

    logistic_result = 1/(1 + np.exp(-(np.dot(beta_vector,X_i_colum_vector))))
    
    return logistic_result


(predictor_matrix_numpy, response_vector) = LRP.data_to_matrix_function()
number_of_iteration = float(input("Iteration number: "))
X = np.array(predictor_matrix_numpy)        # X = [1,X1,X2,X3,...,X23] the decision matrix
Y = np.array(response_vector)               # Y is the response vector
print(X)
print(Y)
print('X',': the main decision matrix with dimensions ',X.shape[0],'x',X.shape[1])
print('Y',': the response vector with column size',Y.shape[0])

Beta_coefficients = np.random.rand(1000,2000,24)      # B = [B0,B1,B2,B3,...B23], unknown random coefficients which
Beta_hat = np.zeros(24)
print(Beta_coefficients)                    # will be updated in gradient ascent algorithm.

"""
column1 = X[0,:]
vector2 = np.transpose(Beta_coefficients)
result = 1 / 1 + np.exp(-1*np.dot(Beta_coefficients, column1))
print(result) 
"""

Beta_hat = gradient_ascent(X,Y,Beta_coefficients,number_of_iteration)
print(Beta_hat)

"""
#W matrix generation:
W = np.eye(X.shape[0])
for i in range(0, X.shape[0]):
    W[i,i] = logistic_function(Beta_coefficients,X[i,:])*(1-logistic_function(Beta_coefficients,X[i,:]))

print(np.dot(np.dot(np.transpose(X),W),X))

"""


