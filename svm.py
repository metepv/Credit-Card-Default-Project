"""
SUPPORT VECTOR MACHINE
WORK IN PROGRESS.............
"""
import numpy as np
import random
import math
import Logistic_Regression_Preprocess as LRP

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

def gradient_descent(X,Y,C,learning_rate,tolerance,iterations):
    
    W=np.zeros(len(X[0]))
    
    for i in range (0,iterations):
        gradient=np.zeros(len(X[0]))
        
        for k in range(0,len(X)):
            if Y[k] * np.dot(W, X[k])<1:

                gradient+=C*Y[k]*X[k]

        
        W=W-learning_rate*gradient
        if np.all(np.abs(gradient*learning_rate) <= tolerance):   
            break
        
    
    return W

def convert_Y(Y_Train):
    for k in range (len(Y_Train)):
        if Y_Train[k]==0:
            Y_Train[k]=-1
    return Y_Train

def Support_Vector_Machine(X_Train,X_Test,Y_Train,iterations,tolerance,learning_rate,C):

    Y_Train2=convert_Y(Y_Train)

    print(np.mean(X_Train))
    X_Train = np.insert(X_Train,0,2,axis = 1)
    X_Test = np.insert(X_Test,0,2,axis = 1)


    W=gradient_descent(X_Train,Y_Train2,C,learning_rate,tolerance,iterations)

    training_results=[]
    test_results=[]
    
    


 #9000000000000000
    for X1 in X_Train:
        if np.dot(W,X1)>20000:
            #print(np.dot(W,X1))
            training_results.append(1)
        else:
            training_results.append(0)

    for X2 in X_Test:
        if np.dot(W,X2)>20000:
            test_results.append(1)
        else:
            test_results.append(0)
        

    return training_results,test_results

iterations=200
tolerance=5e-06
learning_rate=0.0002
C=45

(X, Y) = LRP.data_to_matrix_function()

X=(X-(np.mean(X)))/np.std(X)
 
X_Train=X[0:1000]
X_Test=X[1000:2000]
Y_Train=Y[0:1000]
Y_Test=Y[1000:2000]

training_results,test_results=Support_Vector_Machine(X_Train,X_Test,Y_Train,iterations,tolerance,learning_rate,C)



summm=0
summm2=0

for k in range(len(test_results)):
    if test_results[k]==Y_Test[k]:
        summm2+=1
        if Y_Test[k]==1:
            summm+=1

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0

for i in range(0,len(test_results)):
    if (test_results[i] == Y_Test[i]) and (Y_Test[i] == 1):
        true_positive += 1
    elif (test_results[i] == Y_Test[i]) and (Y_Test[i] == 0):
        true_negative += 1
    elif (test_results[i] != Y_Test[i]) and (Y_Test[i] == 1):
        false_positive += 1
    elif (test_results[i] != Y_Test[i]) and (Y_Test[i] == 0):
        false_negative += 1
        

#print(training_results)

#print(test_results)
#print(Y_Test)
#print(summm2/len(test_results))
#print(summm2/summm)

accuracy = ((true_positive+true_negative)/len(test_results))

print("False Negatives:",false_negative)
print("False Positives:",false_positive)

print("True Negatives:",true_negative)
print("True Positives:",true_positive)
print("Accuracy:",accuracy)