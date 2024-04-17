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
    #decision_matrix_function(raw_data, column_size)
    
    predictor_matrix = [] 
    response_vector = []
     
    for i in range(1,column_size+2):                        #first two line are predictor names.
        
        row_vector_str = raw_data.readline()
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
    
    response_vector_org = response_vector

    #normalization
    #predictor_matrix=(predictor_matrix-(np.mean(predictor_matrix)))/np.std(predictor_matrix)
    #response_vector=(response_vector-(np.mean(response_vector)))/np.std(response_vector)

    return predictor_matrix, response_vector,response_vector_org

"""
(predictor_matrix_numpy, response_vector) = data_to_matrix_function()
X = np.array(predictor_matrix_numpy)
Y = np.array(response_vector)
dimensions_X = X.shape
dimensions_Y = Y.shape
print(X)
print(Y)

"""

"""
print('X',': the main decision matrix with dimensions ',dimensions_X[0],'x',dimensions_X[1])
print('Y',': the response vector with column size',dimensions_Y[0])
"""

