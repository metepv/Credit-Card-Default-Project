"""

EEE 485 PROJECT: Credit Card Default

Module 1:This module imports raw data in a specified size and generates 
         main decision matrix X.

"""

import numpy as np

def decision_matrix_function(raw_data,column_size):
    #this function will take the raw data and column size
    #and will return the main X decision matrix as an output.
    
    X = [] #main decision matrix which contains predictors.
     
    for i in range(1,column_size+23):                        #first two line are predictor names.
        
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
            column_vector_lst[-1] = column_vector_lst[0:-2]

            column_vector_int_lst = []                                                 
            for i2 in range(0,len(column_vector_lst)-1):
                 column_vector_int_lst.append(float(column_vector_lst[i2]))

            X.append(column_vector_int_lst)
        else:
            continue
        
    return X

raw_data = open("UCI_Credit_Card.csv",'r')
#the raw data obtained from kaggle.com as below:
#https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download

column_size = int(input("Specified column size to be imported: "))
#column size will be input for model training with X which will have variying column size
#there are 23 number of fixed features.
#X1,X2,X3,X4,...,X23.
#decision_matrix_function(raw_data, column_size)

matrix  = decision_matrix_function(raw_data, column_size) #Here the ID numbers of customers present in the matrix.
matrix_numpy = np.array(matrix)
X = matrix_numpy[:, 1:] #X = [X1,X2,X3,X4,...,X23]
print(X)
dimensions = X.shape
print("X",":","matrix with dimensions ",dimensions[0],"x",dimensions[1],)

