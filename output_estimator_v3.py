from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from scipy.sparse import csr_matrix
from utils.utils import sparse2tensor, tensor2sparse, minibatch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def compute_HVP(loss_grad, parameters, vec):
    """
    计算Hessian向量积（HVP）。这里参数`loss_grad`是对原始损失函数关于参数的梯度，
    `parameters` 是模型参数的列表，`vec` 是我们要与Hessian相乘的向量。
    """
    hvp = torch.autograd.grad(outputs=loss_grad,inputs=parameters,grad_outputs=vec,retain_graph=True,allow_unused=True)
    return hvp

def Outcome_Estimater(z, model, target_user,target_user_id, n_items,n_users,n_fakes,target_item):
    # 设置模型为训练模式
    sample_data=z
    sample_users_id=[cnt for cnt in range(n_users,n_users+n_fakes)]
    sample_data_csr=csr_matrix(sample_data)
    sample_data_tensor=sparse2tensor(sample_data_csr)
    sample_data_tensor=sample_data_tensor.to("cuda")
    
    target_user_csr=csr_matrix(target_user)
    target_user_tensor=sparse2tensor(target_user_csr)
    target_user_tensor=target_user_tensor.to("cuda")
    
    current_model=model
    # 确定模型阶段是训练模式
    current_model_nn=current_model.net.to("cuda")
    current_model_nn.train()

    # 删除旧的梯度
    current_model_nn.zero_grad()

    # 计算关于测试样本的损失梯度：∇_θ L(z_test; θ_hat)
    pred_test = current_model_nn(user_id=target_user_id)
    loss_test = mse_loss(target_user_tensor,pred_test,current_model.weight_alpha)
    loss_test=loss_test.mean()
    grad_test = torch.autograd.grad(loss_test, current_model_nn.parameters(), create_graph=True)

    # 计算关于所有样本的损失函数的梯度：∇_θ L(sample_data; θ_hat)
    pred_sample = current_model_nn(user_id=sample_users_id)
    loss_sample = mse_loss(sample_data_tensor,pred_sample,current_model.weight_alpha)
    loss_sample=loss_sample.mean()
    grad_sample = torch.autograd.grad(loss_sample, current_model_nn.parameters(), create_graph=True)

    # 初始Hessian向量积：v = ∇_θ L(z_test; θ_hat)
    v = grad_test

    # 计算Hessian向量积：HVP = ∇_θ² L(sample_data; θ_hat) v
    #hvp=[]
    #for grad in grad_sample:
    hvp = compute_HVP(grad_sample, current_model_nn.parameters(), v)
    #hvp.append(hvp_single)

    # 计算影响函数的估计值：-H⁻¹ ∇_θ L(z_j, θ_hat)
    # 这里以一阶Taylors近似代替Hessian的逆
    influence = -sum(torch.sum(h * v_i) for h, v_i in zip(hvp, v))

    return influence.item()  # 返回单个数值

def compute_influence_on_test_loss(model, test_data, train_data, loss_function):
    """
    计算在测试数据上的损失对训练数据的影响。

    model: 训练好的模型。
    test_data: 用于测试的数据。
    train_data: 训练数据，用于计算梯度和HVP。
    loss_function: 损失函数。
    """
    # 首先计算在测试数据上的损失函数关于参数的梯度
    test_loss = loss_function(model(test_data), test_data)
    test_loss_grad = torch.autograd.grad(test_loss, model.parameters(), create_graph=True)

    influences = []
    for data in train_data:
        # 计算训练数据上的损失函数
        train_loss = loss_function(model(data), data)
        # 计算损失函数对参数的梯度
        train_loss_grad = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
        # 计算Hessian向量积
        hvp = compute_HVP(train_loss_grad, model.parameters(), test_loss_grad)
        # 计算负的Hessian向量积，即作为损失函数对测试样本影响的近似
        influence = -sum(torch.sum(a * b) for a, b in zip(hvp, train_loss_grad))
        influences.append(influence)

    return influences

def compute_gradients(model, loss_function, data):
    # 前向传播计算损失
    predictions = model(data)
    loss = loss_function(data, predictions)
    # 反向传播计算参数的梯度
    loss.backward()
