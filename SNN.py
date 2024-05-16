#EEE 485: Statsitical Learning and Data Analytics Term Project
#Shallow Neural Network Algorithm Implementation

import numpy as np
import math
import random
import pandas as pd


dt = pd.read_csv('UCI_Credit_Card.csv')

dt.columns = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","y"]
X = np.array(dt)[:,1:24]
Y = np.array(dt)[:,24]


columns=[0,4,11,12,13,14,15,16,17,18,19,20,21,22]
for xi in columns:

    mean=np.mean(X[:,xi])
    std=np.std(X[:,xi])
    X[:,xi]=(X[:,xi]-mean)/std
    
 
X_Train=X[0:50]
X_Test=X[50:100]
Y_Train=Y[0:50]
Y_Test=Y[50:100]

# dt = pd.read_csv('CKD_Preprocessed.csv')

# #shuffle all rows
# dt = dt.sample(frac = 1)

# dt.columns = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","y"]
# matrix = np.array(dt)
# X=matrix[:,0:24]
# Y= matrix[:,24]
"""
shuffle_list =  list(zip(X,Y))
random.shuffle(shuffle_list)
X,Y=zip(*shuffle_list)
Y=np.array(Y)
X=np.array(X)
"""
# columns=[0,1,2,5,6,7,8,9,10,11,12,13]
# for column in columns:
#     X[:,column]=X[:,column]-np.mean(X[:,column])
#     X[:,column]=X[:,column]/np.std(X[:,column])
#     #print("okan bok kokan")


# X_Train = (X[0:200])
# X_Test = (X[200:400])
# Y_Train = Y[0:200]
# Y_Test = Y[200:400]


def sigmoid(z):
    return 1/(1+np.exp(-1*z))

def sigmoid_derivative(z):
    return (1-sigmoid(z))*sigmoid(z)

def pred(X,input_hidden_weights,hidden_output_weights):
    return 1 if forward_propagation(X,input_hidden_weights,hidden_output_weights)[0][0][0]>=0.5 else 0

def mse_loss(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def init_weights(sizeof_input, sizeof_output):
    
    #Xavier initialization for weights of a neural network layer.
    weights = np.random.normal(0, math.sqrt(2/(sizeof_input+sizeof_output)), (sizeof_input, sizeof_output))
    return weights

input_size = 23
hidden_layer_size = 15
output_size = 1

# Initialize weights
input_hidden_weights = init_weights(input_size, hidden_layer_size)
hidden_output_weights = init_weights(hidden_layer_size, output_size)

def forward_propagation(X,input_hidden_weights,hidden_output_weights):
    results_hidden_layer=[]
    final_out=[]
    
    for k in range (len(X)):
        input_hidden_layer=0
        for i in range(len(X[0])):
            input_hidden_layer+=np.dot(X[k][i],(input_hidden_weights[i]))
        result_hidden_layer=(sigmoid(input_hidden_layer))
        #sigmoid is for activation
        results_hidden_layer.append(result_hidden_layer)
        input_output=np.dot(result_hidden_layer,hidden_output_weights)

        final_out.append(sigmoid(input_output))
    #return final_out,result_hidden_layer
    final_out=np.array(final_out)
    
    results_hidden_layer=np.array(results_hidden_layer)
    
    return final_out,results_hidden_layer

def average_of_lists(*lists):
    # Initialize a list to store the averages
    averages = []

    # Iterate through the lists
    for values in zip(*lists):
        # Calculate the average of corresponding elements
        avg = sum(values) / len(values)
        averages.append(avg)

    return averages


def backpropagation(X, y_true, input_hidden_weights, hidden_output_weights, learning_rate, epochs):
    
    losses=[]
    for epoch in range(epochs):
        # Calculate output
        (y_pred,hidden_layer_output) = forward_propagation(X, input_hidden_weights,hidden_output_weights)
        
        # Calculate loss
        loss = mse_loss(y_true, y_pred)
        losses.append(loss)
        if len(losses)>2:
            if losses[-1]>losses[-2]:
                return hidden_output_weights,input_hidden_weights
        # Backrpropagation
        output_error = y_true - y_pred
        output_delta = output_error * sigmoid_derivative(y_pred)
        #transpose may be needed
       
        hidden_error = np.dot(output_delta,hidden_output_weights.T)
       
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
        #print(hidden_layer_output)
        # Update weights (transpose may be needed)
        
        hidden_output_weights=np.add(hidden_output_weights, np.dot(hidden_layer_output.T,output_delta) * learning_rate)
        
        input_hidden_weights=np.add(input_hidden_weights, np.dot(X.T,hidden_delta) * learning_rate)
        newloss=loss
        
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss}")
    
    return hidden_output_weights,input_hidden_weights

#find weights avrg
learning_rate=0.002
epochs=1000
hidden_output_weights_list=[]
input_hidden_weights_list=[]

for k in range (len(X_Train)):
    X_Train_1=np.array([X_Train[k]])
    Y_Train_1=np.array([Y_Train[k]])
    hidden_output_weights,input_hidden_weights=backpropagation(X_Train_1, Y_Train_1, input_hidden_weights, hidden_output_weights, learning_rate, epochs)
    hidden_output_weights_list.append(hidden_output_weights)
    input_hidden_weights_list.append(input_hidden_weights)

def sum_and_average_arrays(*arrays):
    # Convert the list of arrays into a numpy array
    stacked_arrays = np.stack(arrays, axis=0)
    # Sum along the first axis
    summed_array = np.sum(stacked_arrays, axis=0)
    # Calculate the average
    average_array = summed_array / len(arrays)
    
    return average_array


def Shallow_Neural_Network(X_Test,input_hidden_weights,hidden_output_weights):

    test_predictions = []
    for r in range(len(X_Test)):
        prediction = forward_propagation(X_Test,input_hidden_weights,hidden_output_weights)[0][0][0]
        print(prediction)
        if prediction >0.3:
            test_predictions.append(1)
        else:
            test_predictions.append(0)
    return test_predictions


resultlist=[]
for k in range (len(X_Test)):
    X_Train_1=np.array([X_Train[k]])
    #X_Train_2=np.array([X_Train[1]])
    X_Test_1=np.array([X_Test[k]])
    Y_Train_1=np.array([Y_Train[k]])
    results=Shallow_Neural_Network(X_Train_1,sum_and_average_arrays(*input_hidden_weights_list),sum_and_average_arrays(*hidden_output_weights_list))
    #results=Shallow_Neural_Network(X_Test_1,input_hidden_weights_list[k],hidden_output_weights_list[k])
    resultlist.append(results)


correct=0
true_negative=0
true_positive=0
false_negative=0
false_positive=0
negative=0
positive=0


for e in range(0,len(Y_Train)):
    if Y_Train[e]==1:
        positive+=1
    else:
        negative+=1

for i in range(0,len(resultlist)):
        #print(resultlist[i][0],Y_Train[i])
        if (resultlist[i][0] == Y_Train[i]) and (resultlist[i][0] == 1):
            true_positive += 1
        elif (resultlist[i][0] == Y_Train[i]) and (resultlist[i][0] == 0):
            true_negative += 1
        elif (resultlist[i][0] != Y_Train[i]) and (resultlist[i][0] == 1):
            false_positive += 1
        elif (resultlist[i][0] != Y_Train[i]) and (resultlist[i][0] == 0):
            false_negative += 1
            

print(resultlist)

accuracy=(true_negative+true_positive)/len(Y_Test)

print(accuracy)
print("Negative:",negative,"Positive:",positive,"True Negatives:",true_negative,"True Positives:",true_positive,"False Negatives:",false_negative,"False Positives:",false_positive)
#print(hidden_output_weights)
#print(forward_propagation(X_Train,input_hidden_weights,hidden_output_weights)[0][0][0])
