"""

EEE 485 PROJECT: Credit Card Default

Module 2:This module takes the decision matrix X and makes prediction
         with logistic regression.

"""
import numpy as np
import Module1 as md1

(predictor_matrix_numpy, response_vector) = md1.data_to_matrix_function()
X = np.array(predictor_matrix_numpy)
Y = np.array(response_vector)
