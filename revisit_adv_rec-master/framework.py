import time
import argparse
import importlib

from bunch import Bunch
import numpy as np
import random
from scipy.sparse import csr_matrix

from data.data_loader import DataLoader
from utils.utils import set_seed, stack_csrdata, load_fake_data
from utils.utils import save_fake_data

from adversary_example_generator_v2 import adversary_pattern_generator


def init(args,user_cnt,item_cnt,train_data,test_data):
    attack_eval_args = Bunch(args.attack_eval_args)
    for victim_args in attack_eval_args.victims:
        print(victim_args)
        victim_args = Bunch(victim_args)

        trainer_class = victim_args.model["trainer_class"]
        trainer = trainer_class(n_users=user_cnt,
                                n_items=item_cnt,
                                args=victim_args)
        trainer.fit(train_data, test_data)
    return trainer
        

if __name__ == "__main__":
    seed=time.time()
    set_seed(int(seed),cuda=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="evaluate_attack_args")
    config = parser.parse_args()
    args = importlib.import_module(config.config_file)
    
    print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()
    
    mid_model=init(args,n_users,n_items,train_data,test_data)
    
    target_item=[1,2,3,4,5]
    target_user=[6,7,8,9,10]
    adversary_pattern,item_clusters=adversary_pattern_generator(mid_model,train_data,target_item,target_user)

    fake_user_amount=3    
    fake_users=np.zeros((fake_user_amount,n_items))

    for i in range(fake_user_amount):
        for j in range(len(adversary_pattern)):
            group_id=adversary_pattern[j]
            random_index = random.randint(0, len(item_clusters[group_id]) - 1)
            sampled_item = item_clusters[group_id][random_index]
            fake_users[i][sampled_item]=1
    fake_users_record=csr_matrix(fake_users)
    save_fake_data(fake_users_record, path='./generated_fake_users')