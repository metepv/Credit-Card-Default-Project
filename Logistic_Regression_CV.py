"""
EEE 485 PROJECT: Credit Card Default
Module 3: K-Fold Cross-Validation for Logistic Regression
          K = 10

"""

import numpy as np
import Logistic_Regression_Preprocess as LRP

all_data, all_response = LRP.data_to_matrix_function() 
all_data = np.array(all_data)
all_response = np.array(all_response)

fold_size = int(input("fold size: "))
lambda_parameter_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
j_lst = np.random.choice(np.arange(0, 9+1), size=10, replace=False)
f_matrix = []
count = 0

for i in range(0,10):

    jth_fold = all_data[j_lst[i]*fold_size:j_lst[i]*fold_size + fold_size,:]
    f_matrix.append(jth_fold)

while count != 10:
    
    count += 1
    


