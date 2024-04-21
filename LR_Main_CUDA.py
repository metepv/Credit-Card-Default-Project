"""
EEE 485 PROJECT: Credit Card Default

Module 2:This module takes the decision matrix X and the response vector Y to make prediction
         with logistic regression.
"""
import torch
import time
import LR_Pre as LR_Pre

def Newton_Raphson():

    (predictor_matrix_numpy, response_vector) = LR_Pre.data_to_matrix_function()
    iteration_number = int(input("Iteration number: "))
    X = torch.FloatTensor(predictor_matrix_numpy).cuda()        # X = [1,X1,X2,X3,...,X23] the decision matrix
    Y = torch.FloatTensor(response_vector).cuda()               # Y is the response vector
    print(X.device)

    Beta_coefficients = torch.ones(24).cuda()*0.5          # B = [B0,B1,B2,B3,...B23], unknown random coefficients which

    Beta_new = torch.zeros(24).cuda()
    Beta_old = Beta_coefficients

    row_size = X.shape[0]

    #Pi vector generation:
    Pi_vector = []                        
    for i2 in range(0, row_size):
        Pi_vector.append(torch.tensor(logistic_function(Beta_old,X[i2,:])))
    Pi_vector = torch.stack(Pi_vector).cuda()

    #W matrix generation:
    W = torch.eye(X.shape[0]).cuda()
    for i in range(0, X.shape[0]):
        W[i,i] = logistic_function(Beta_old,X[i,:])*(1-logistic_function(Beta_old,X[i,:]))

    X_T = torch.transpose(X,0,1).cuda()
    A = torch.matmul(torch.matmul(X_T,W), X).cuda()  #A = X_t*W*X
    A_inverse = torch.inverse(A).cuda()  #A^-1 = (X_t*W*X)^-1

    count = 0

    while count != iteration_number:

        A = torch.matmul(torch.matmul(X_T,W), X).cuda() #A = X_t*W*X
        A_inverse = torch.inverse(A).cuda() #A^-1 = (X_t*W*X)^-1
        C1 = torch.matmul(A_inverse,X_T).cuda()
        C2 = Y - Pi_vector
        C3 = torch.matmul(C1,C2).cuda()
        Beta_new = Beta_old + C3

        #updating matrices and coefficients.
        Beta_old = Beta_new
        
        for i in range(0, X.shape[0]):
            Pi_vector[i] = (logistic_function(Beta_old,X[i,:])) 

        for i in range(0, X.shape[0]):
            W[i,i] = logistic_function(Beta_old,X[i,:])*(1-logistic_function(Beta_old,X[i,:]))
        
        count += 1
        
        print(count)

    return Beta_new
    

def logistic_function(beta_vector,X_i_colum_vector):

    logistic_result = torch.float64
    dot_product = torch.float64
    dot_product = torch.dot(beta_vector,X_i_colum_vector)/10000

    logistic_result = 1/(1 + torch.FloatTensor.exp(-dot_product))
    
    return logistic_result
