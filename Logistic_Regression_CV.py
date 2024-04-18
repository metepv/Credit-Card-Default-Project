"""
EEE 485 PROJECT: Credit Card Default
Module 3: K-Fold Cross-Validation for Logistic Regression
          K = 10

"""
import numpy as np
import Logistic_Regression_Preprocess as LRP
import Logistic_Regression_Main_W as LRM_W

all_data, all_response = LRP.data_to_matrix_function() 
all_data = np.array(all_data)
all_response = np.array(all_response)

fold_size = int(input("fold size: "))
lambda_parameter_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
j_lst = np.random.choice(np.arange(0, 9+1), size=10, replace=False)
f_matrix = []
f_matrix_response = []
fold_index = 0
accuracy_lst =[]

for i in range(0,10):

    jth_fold = all_data[j_lst[i]*fold_size:j_lst[i]*fold_size + fold_size,:]
    jth_fold_response = all_response[j_lst[i]*fold_size:j_lst[i]*fold_size + fold_size]
    f_matrix.append(jth_fold)
    f_matrix_response.append(jth_fold_response)

while fold_index != 10:

    training_accuracy_elements      = [0,0,0,0] #1.TP, 2.TN, 3.FP, 4.FN
    fold_testing_accuracy_elements  = [0,0,0,0] #1.TP, 2.TN, 3.FP, 4.FN

    selected_lambda = lambda_parameter_list[fold_index]
    Y_test = f_matrix_response[fold_index]
    X_test = f_matrix[fold_index]

    #for X_train
    for fold in range(0,10):
        if fold != fold_index:
            X_train = np.vstack(f_matrix[fold])
        else:
            continue
    
    #for Y_train
    Y_train = []
    for fold in range(0,10):
        if fold != fold_index:
            Y_train.extend(f_matrix_response[fold])
        else:
            continue
    
    Beta_hat = LRM_W.gradient_ascent(X_train,Y_train)

    #---------------training accuracy part-----------
    log_estimate_train = []

    for row in range(0, len(X_train)):
        result = LRM_W.logistic_function(Beta_hat, X_train[row])
        if result >= selected_lambda:
            log_estimate_train.append(1)
        else:
            log_estimate_train.append(0)
    
    y = np.array(Y_train)
    log = np.array(log_estimate_train)

    for i in range(0,len(Y_train)):

        if (log_estimate_train[i] == Y_train[i]) and (Y_train[i] == 1):
            training_accuracy_elements[0] += 1
        elif (log_estimate_train[i] == Y_train[i]) and (Y_train[i] == 0):
            training_accuracy_elements[1] += 1
        elif (log_estimate_train[i] != Y_train[i]) and (Y_train[i] == 1):
            training_accuracy_elements[2] += 1
        elif (log_estimate_train[i] != Y_train[i]) and (Y_train[i] == 0):
            training_accuracy_elements[3] += 1
    
    training_accuracy = (training_accuracy_elements[0] + training_accuracy_elements[1])/(all_data.shape[0]-fold_size)

    #---------------testing accuracy part------------
    log_estimate_test = []

    for row in range(0, len(X_train)):
        result = LRM_W.logistic_function(Beta_hat, X_test[row])
        if result >= selected_lambda:
            log_estimate_test.append(1)
        else:
            log_estimate_test.append(0)
    
    y = np.array(Y_test)
    log = np.array(log_estimate_test)

    for i in range(0,len(Y_test)):

        if (log_estimate_train[i] == Y_train[i]) and (Y_train[i] == 1):
            fold_testing_accuracy_elements[0] += 1
        elif (log_estimate_train[i] == Y_train[i]) and (Y_train[i] == 0):
            fold_testing_accuracy_elements[1] += 1
        elif (log_estimate_train[i] != Y_train[i]) and (Y_train[i] == 1):
            fold_testing_accuracy_elements[2] += 1
        elif (log_estimate_train[i] != Y_train[i]) and (Y_train[i] == 0):
            fold_testing_accuracy_elements[3] += 1
    
    test_accuracy = (fold_testing_accuracy_elements[0] + fold_testing_accuracy_elements[1])/(all_data.shape[0]-fold_size)
    
    accuracy_tuple = (training_accuracy,test_accuracy)
    accuracy_lst.append(accuracy_tuple)
    
    fold_index += 1
    


