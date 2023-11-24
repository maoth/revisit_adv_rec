from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from scipy.sparse import csr_matrix
from utils.utils import sparse2tensor, tensor2sparse, minibatch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def calculate_influence_function(model, target_user_data,sample_data_tensor, sample_users_id):
    # gradient= influence of poisoning data to model
    # Calculate the gradient of the loss with respect to the model parameters
    model_nn=model.net.to("cuda")
    model_nn.zero_grad()
    model_nn.eval()
    logits=model_nn(user_id=sample_users_id)
    loss=mse_loss(sample_data_tensor,logits,model.weight_alpha)
    loss=loss.mean()
    grad = torch.autograd.grad(loss, model_nn.parameters(), create_graph=True,retain_graph=True) #9000*128

    # Perform Hessian-vector product approximation
    hvp=[]
    for i in range(len(grad)):
        grad[i].requires_grad_(True)
        hvp.append(torch.autograd.grad(grad[i], model_nn.parameters(), grad_outputs=grad[i],retain_graph=True)) 
        hvp[i]=torch.cat(hvp[i])
    influence = [(hvp[i] / len(target_user_data)) for i in range(len(hvp))]
    del model_nn

    return influence

def calculate_influence_on_prediction(influence, model,target_user_data,target_item,n_users,n_items):
    # target user score of influenced model
    # Calculate the gradient of the prediction scoring function with respect to the parameters
    model_nn=model.net.to("cuda")
    model_nn.zero_grad()
    scores = model_nn(user_id=target_user_data)
    scores=scores[:,target_item-1]
    #scores=scores.mean()
    grad=[]
    influence_on_prediction=[]
    influence_size=list(influence[0].size())
    final_influence=torch.transpose(influence[0],0,1)
    for i in range(len(scores)):
        grad=torch.autograd.grad(scores[i], model_nn.parameters(), create_graph=True)
        grad=torch.cat(grad)

        # Calculate the influence on the prediction scoring function
        influence_result=-torch.matmul(grad, final_influence)
        influence_result=influence_result[n_items:]
        influence_result=influence_result[target_user_data]
        influence_result=influence_result[:,:n_items]
        influence_result=influence_result[:,target_item-1]
        influence_result=torch.flatten(influence_result)
        influence_on_prediction.append(influence_result.detach().cpu())
        del grad
        del influence_result
    del model_nn

    return influence_on_prediction  #target_user*dim
        
def Outcome_Estimater(z,model,target_user,target_user_id,n_items,n_users,n_fakes,target_item):
    current_model=model
    sample_data_list=z
    sample_users_id=[cnt for cnt in range(n_users,n_users+n_fakes)]

    sample_data_csr=csr_matrix(sample_data_list)
    sample_data_tensor=sparse2tensor(sample_data_csr)
    sample_data_tensor=sample_data_tensor.to("cuda")
    target_user_data=target_user.toarray()
    for i in range(len(target_user_data)):
        target_user_data[i][target_item]=1
        target_user_data[i][n_items-1]=1
    target_user_csr=csr_matrix(target_user_data)

    sample_data_csr=csr_matrix(sample_data_list)
    
    influence_j = calculate_influence_function(current_model, target_user_id,sample_data_tensor, sample_users_id)
    influence_on_prediction = calculate_influence_on_prediction(influence_j, current_model,target_user_id,target_item,n_users,n_items)
    
    influence=influence_on_prediction
    del current_model
    del sample_data_tensor

    return influence