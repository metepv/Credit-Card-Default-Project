"""
EEE 485 PROJECT: Credit Card Default

Module 2:This module takes the decision matrix X and the response vector Y to make prediction
         with logistic regression.
"""
import numpy as np
import Logistic_Regression_Preprocess as LRP

(predictor_matrix_numpy, response_vector) = LRP.data_to_matrix_function()
X = np.array(predictor_matrix_numpy)        # X = [1,X1,X2,X3,...,X23] the decision matrix
Y = np.array(response_vector)               # Y is the response vector
print(X)
print(Y)

Beta_coefficients = np.random.rand(24)      # B = [B0,B1,B2,B3,...B23]
print(Beta_coefficients)