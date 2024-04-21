
import torch
import LR_Pre as LRP

design_matrix, response_vector = LRP.data_to_matrix_function()

X = torch.FloatTensor(design_matrix).cuda()
Y = torch.FloatTensor(response_vector).cuda()

print(X.device)

