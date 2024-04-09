"""
EEE 485 PROJECT: Credit Card Default

Module 2:This module takes the decision matrix X and the response vector Y to make prediction
         with logistic regression.
"""
import numpy as np
import Module1 as md1

(predictor_matrix_numpy, response_vector) = md1.data_to_matrix_function()
X = np.array(predictor_matrix_numpy)        # X = [1,X1,X2,X3,...,X23] the decision matrix
Y = np.array(response_vector)               # Y is the response vector
print(X)
print(Y)

Beta_coefficients = np.random.rand(24)
print(Beta_coefficients)