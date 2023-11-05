from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from scipy.sparse import csr_matrix
from utils.utils import sparse2tensor, tensor2sparse, minibatch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def Outcome_Estimater(z,upweight_threshold,model,rec_normal_result,n_items,n_user,target_user,target_item):
    sample_data_list=np.array(z)
    sample_data=np.zeros((1,n_items))
    for i in range(sample_data_list.size):
        sample_data[0][sample_data_list[i]]=1
    sample_data_csr=csr_matrix(sample_data)
       
    delta=upweight_threshold
    current_model=model
    normal_users_result=rec_normal_result
    rec_result=current_model.recommend(sample_data_csr,n_items)
    rec_result=np.concatenate((normal_users_result,rec_result))
    rank_of_items=[np.where(row==target_item)[0] for row in rec_result]
    rank_of_items=np.asarray(rank_of_items)
    rank_of_items=np.reshape(rank_of_items,(1,len(rank_of_items)))[0]
    real_target_users=[]
    for i in range(len(rank_of_items)):
        if rank_of_items[i]>20:
            real_target_users.append((i,rank_of_items[i]))
    real_target_users=sorted(real_target_users,key=lambda x:x[1])
    real_target_users=real_target_users[:int(n_user*0.1)]
    real_target_users_id=[ele[1] for ele in real_target_users]
    
    current_model_nn=current_model.net.to("cuda")
    current_model_nn.eval()
    logits=current_model_nn(user_id=real_target_users_id)

    sample_data_tensor=sparse2tensor(sample_data_csr)
    sample_data_tensor=sample_data_tensor.to("cuda")

    loss=mse_loss(sample_data_tensor,logits,current_model.weight_alpha)
    loss=torch.sum(loss)

    grad = torch.autograd.grad(loss, current_model_nn.params, create_graph=True)

    sample_data_tensor=torch.transpose(sample_data_tensor,0,1)
    grad_dot_Z = torch.sum(grad[0] * sample_data_tensor)

    hvp = torch.autograd.grad(grad_dot_Z, current_model_nn.params)

    H_matrix_inverse=hvp[0]
    gradients=grad[0]
    
    rank = np.where(rec_result == sample_data_list[sample_data_list.size-1])[1][0]
    out_score=rank*(-1)
      
    influence=(-1)* gradients * out_score * H_matrix_inverse * gradients * loss
    influence=influence/1e10
    
    del current_model_nn
    del sample_data_tensor

    return influence.float()