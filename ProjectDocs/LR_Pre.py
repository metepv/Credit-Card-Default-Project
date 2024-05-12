"""
EEE 485 PROJECT: Credit Card Default

Module 1:This module imports raw data in a specified size and generates 
         main decision matrix X and response vector Y.
"""

import numpy as np

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
    
    predictor_matrix = [] 
    response_vector = []
     
    for i in range(1,column_size+2): #first two line are predictor names.
        
        row_vector_str = raw_data.readline()

        if row_vector_str[0].isdigit():
            row_vector_lst = row_vector_str.split(',')
            row_vector_lst[-1] = row_vector_lst[-1][0:1]
            response_vector.append(int(row_vector_lst[-1]))

            row_vector_float_lst = [1]                                                 
            for i2 in range(1,len(row_vector_lst)-1): # last value is the y value of the regression.
                 row_vector_float_lst.append(float(row_vector_lst[i2]))
            
            predictor_matrix.append(row_vector_float_lst)
        else:
            continue

    return predictor_matrix, response_vector


