
import numpy as np
import time as time
import matplotlib.pyplot as plt
import Logistic_Regression_Main_W as LRP_W

start_time = time.time()

(Beta_hat,X,Y,Y_un)= LRP_W.gradient_ascent()

n = len(X)
lambda_parameter = 0.5
log_estimate = []

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0

for row in range(0, len(X)):
    result = LRP_W.logistic_function(Beta_hat, X[row])
    if result >= lambda_parameter:
        log_estimate.append(1)
    else:
        log_estimate.append(0)

y = np.array(Y_un)
log = np.array(log_estimate)

for i in range(0,len(Y_un)):
    if (log_estimate[i] == Y_un[i]) and (Y_un[i] == 1):
        true_positive += 1
    elif (log_estimate[i] == Y_un[i]) and (Y_un[i] == 0):
        true_negative += 1
    elif (log_estimate[i] != Y_un[i]) and (Y_un[i] == 1):
        false_positive += 1
    elif (log_estimate[i] != Y_un[i]) and (Y_un[i] == 0):
        false_negative += 1

print(y)
print(log)

accuracy = ((true_negative + true_positive)/n)*100
print(accuracy)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

