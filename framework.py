import time
import argparse
import importlib
import os

from bunch import Bunch
import numpy as np
import random
from scipy.sparse import csr_matrix

from data.data_loader import DataLoader
from utils.utils import set_seed, stack_csrdata, load_fake_data
from utils.utils import save_fake_data

from adversary_example_generator_v2 import adversary_pattern_generator      

def init_model(args,user_cnt,item_cnt,train_data,test_data):
    simulator_args = Bunch(args.surrogate)
    simulator_args = Bunch(simulator_args)

    trainer_class = simulator_args.model["trainer_class"]
    trainer = trainer_class(n_users=user_cnt,
                            n_items=item_cnt,
                            args=simulator_args)
    trainer.fit(train_data, test_data)
    return trainer

def DQN_attack(n_users,n_items,train_data,test_data,args):
    seed=time.time()
    set_seed(int(seed),cuda=True)
    mid_model=init_model(args,n_users,n_items,train_data,test_data)
    
    target_item=args.target_items
    target_user=list(range(n_users))
    adversary_pattern,item_clusters=adversary_pattern_generator(mid_model,train_data,test_data,target_item,target_user,args)

    fake_user_amount=3    
    fake_users=np.zeros((fake_user_amount,n_items))

    for i in range(fake_user_amount):
        for j in range(len(adversary_pattern)):
            group_id=adversary_pattern[j]
            random_index = random.randint(0, len(item_clusters[group_id]) - 1)
            sampled_item = item_clusters[group_id][random_index]
            fake_users[i][sampled_item]=1
    fake_users_record=csr_matrix(fake_users)
    fake_data_path = os.path.join(
            args.output_dir,
            "_".join(["DQN", "fake_data", "best",args.tag]))
    save_fake_data(fake_users_record, path=fake_data_path)