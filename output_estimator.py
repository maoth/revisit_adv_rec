from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from scipy.sparse import csr_matrix
from utils.utils import sparse2tensor, tensor2sparse, minibatch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def hvp(y, x, v):
    Hv=[]
    gradient = torch.autograd.grad(y, x,retain_graph=True)
    gradient[0].requires_grad_(True)
    gradient[1].requires_grad_(True)
    Hv.append(torch.autograd.grad(gradient[0], x[0], grad_outputs=v[0],allow_unused=True))
    Hv.append(torch.autograd.grad(gradient[1], x[1], grad_outputs=v[1],allow_unused=True))
    print(Hv)
    return Hv
       
def Outcome_Estimater(z,upweight_threshold,train_data,model,n_items,n_users,n_fakes,target_user,target_item):
    current_model=model
    sample_data_list=z
    sample_users_id=[cnt for cnt in range(n_users,n_users+n_fakes)]
    target_user_data=train_data[target_user].toarray()
    for i in range(len(target_user_data)):
        target_user_data[i][target_item]=1
        target_user_data[i][n_items-1]=1    

    sample_data_csr=csr_matrix(sample_data_list)
    target_user_csr=csr_matrix(target_user_data)


    rec_result=current_model.recommend(target_user_csr,n_items)
       
    delta=upweight_threshold
    
    current_model_nn=current_model.net.to("cuda")
    current_model_nn.eval()
    logits=current_model_nn(user_id=sample_users_id)

    sample_data_tensor=sparse2tensor(sample_data_csr)
    sample_data_tensor=sample_data_tensor.to("cuda")
    
    loss=mse_loss(sample_data_tensor,logits,current_model.weight_alpha)
    loss=loss.mean()
    loss.backward(retain_graph=True)
    
    delta_theta = [param.grad.view(-1) for param in current_model_nn.parameters()]
    network_parameters=list(current_model_nn.parameters())
    delta_theta[0]=torch.reshape(delta_theta[0],(n_items,current_model.dim))
    delta_theta[1]=torch.reshape(delta_theta[1],(n_users+n_fakes,current_model.dim))
    current_model_nn.zero_grad()
    hessian_vector_product= hvp(loss, current_model_nn.params, delta_theta)
    for Hv in hessian_vector_product:
        print(Hv)
    hessian_vector_product = [Hv.view(-1) for Hv in hessian_vector_product]  # Flatten the HVP result
    hessian_vector_product = torch.cat(hessian_vector_product)
    delta_theta=torch.flatten(delta_theta)
    hvp_result = torch.dot(hessian_vector_product, delta_theta)
    
    grad = torch.autograd.grad(loss, current_model_nn.params, create_graph=True)

    rank = np.where(rec_result == target_item)
    print(rank)
    out_score=rank*(-1)
    
    print(out_score,len(out_score))
    print(grad_tensor.size())
    print(hvp_result.size())
      
    influence=grad_tensor * out_score * hvp_result
    influence=influence/1e10
    
    del current_model_nn
    del sample_data_tensor

    return influence.float()