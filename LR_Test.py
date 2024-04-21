
import torch
import time as time
import LR_Main_CUDA as LRM_CUDA

start_time = time.time()

(Beta_hat,X,Y)= LRM_CUDA.Newton_Raphson()

n = len(X)
lambda_parameter = 0.6
log_estimate = []

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0

for row in range(0, len(X)):
    result = LRM_CUDA.logistic_function(Beta_hat, X[row])
    if result >= lambda_parameter:
        log_estimate.append(1)
    else:
        log_estimate.append(0)

y = torch.FloatTensor.array(Y)
log = torch.FloatTensor.array(log_estimate)

for i in range(0,len(Y)):
    if (log_estimate[i] == Y[i]) and (Y[i] == 1):
        true_positive += 1
    elif (log_estimate[i] == Y[i]) and (Y[i] == 0):
        true_negative += 1
    elif (log_estimate[i] != Y[i]) and (Y[i] == 1):
        false_positive += 1
    elif (log_estimate[i] != Y[i]) and (Y[i] == 0):
        false_negative += 1

print(y)
print(log)

accuracy = ((true_negative + true_positive)/n)*100
print(accuracy)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

