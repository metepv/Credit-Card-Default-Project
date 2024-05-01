import numpy as np
import time as time
import LR_Pre as LRP
import LR_Main as LRM

start_time = time.time()
(design_matrix,response) = LRP.data_to_matrix_function()
X = np.array(design_matrix)
Y = np.array(response)
iteration = int(input("Iteration: "))
(Beta_hat)= LRM.Newton_Raphson(X,Y,iteration)

n = len(X)
lambda_parameter = 0.5
log_estimate = []

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0

for row in range(0, len(X)):
    result = LRM.logistic_function(Beta_hat, X[row])
    if result >= lambda_parameter:
        log_estimate.append(1)
    else:
        log_estimate.append(0)

y = np.array(Y)
log = np.array(log_estimate)

for i in range(0,len(Y)):
    if (log_estimate[i] == Y[i]) and (Y[i] == 1):
        true_positive += 1
    elif (log_estimate[i] == Y[i]) and (Y[i] == 0):
        true_negative += 1
    elif (log_estimate[i] != Y[i]) and (Y[i] == 1):
        false_positive += 1
    elif (log_estimate[i] != Y[i]) and (Y[i] == 0):
        false_negative += 1

print("TP: ",true_positive,"TN: ",true_negative,"FP: ",false_positive,"FN: ",false_negative)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

