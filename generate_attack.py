import argparse
import importlib
import sys
import numpy as np
import time

from bunch import Bunch

from data.data_loader import DataLoader
from utils.utils import set_seed, sample_target_items
from framework import DQN_attack

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="generate_attack_args")
parser.add_argument("-dataset", type=str)
parser.add_argument("-att_type", type=str)
parser.add_argument("-ratio", type=float)
parser.add_argument("-pop",type=str)
parser.add_argument("-unroll",type=int)
parser.add_argument("-tag",type=str)
config = parser.parse_args()


def generate(args,config):
    # Load data.
    print(config)
    args.data_path="./data/"+config.dataset
    print("dataset={}".format(args.data_path))
    if config.att_type=="WRMF_ALS":
        args.attack_gen_args=args.attack_gen_args_wrmf_als 
    elif config.att_type=="WRMF_SGD":
        args.attack_gen_args=args.attack_gen_args_wrmf_sgd
    elif config.att_type=="RandFilter":
        args.attack_gen_args=args.attack_gen_args_randfilter
    elif config.att_type=="DQN":
        args.attack_gen_args=args.attack_gen_args_dqn
    else:
        raise Error("Attack method error.")
    print("attack method={}".format(args.attack_gen_args))
    print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))

    # Train & evaluate adversarial users.
    attack_gen_args = Bunch(args.attack_gen_args)
    print("surrogate model={}".format(attack_gen_args.surrogate))
    if config.pop=="head":
        attack_gen_args.target_item_popularity="head"
    elif config.pop=="upper":
        attack_gen_args.target_item_popularity="upper_torso"
    elif config.pop=="lower":
        attack_gen_args.target_item_popularity="lower_torso"
    elif config.pop=="tail":
        attack_gen_args.target_item_popularity="tail"
    else:
        raise Error("Popularity type error.")

    attack_gen_args.n_fakes=config.ratio/100
    attack_gen_args.unroll_steps=config.unroll
    attack_gen_args.tag=config.tag
    attack_gen_args.n_fake_users = int(n_users * attack_gen_args.n_fakes)
    data_loader.n_items=n_items+1
    data_loader.item2id[n_items]=n_items
    for i in range(attack_gen_args.n_fake_users):
        data_loader.user2id[n_users+i]=n_users+i
    n_users, n_items=data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()
    print("popularity={}, fake user ratio={}, fake users={}, unroll epochs={}".format(attack_gen_args.target_item_popularity,attack_gen_args.n_fakes,attack_gen_args.n_fake_users, attack_gen_args.unroll_steps))
    target_items = sample_target_items(
        train_data,
        n_samples=attack_gen_args.n_target_items,
        popularity=attack_gen_args.target_item_popularity,
        use_fix=attack_gen_args.use_fixed_target_item,
        output_dir=attack_gen_args.output_dir,
        tag=attack_gen_args.tag
    )
    attack_gen_args.target_items = target_items
    print(attack_gen_args.target_items)

    if config.att_type=="DQN":
        DQN_attack(n_users=n_users,n_items=n_items,train_data=train_data,test_data=test_data,args=attack_gen_args)
    else:
        adv_trainer_class = attack_gen_args.trainer_class
        adv_trainer = adv_trainer_class(n_users=n_users,
                                        n_items=n_items,
                                        args=attack_gen_args)
        adv_trainer.fit(train_data, test_data)

def evaluate(args,config):
    print(args)
    print(config)

    args.data_path="./data/"+config.dataset
    print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()

    if config.pop=="head":
        popularity="head"
    elif config.pop=="upper":
        popularity="upper_torso"
    elif config.pop=="lower":
        popularity="lower_torso"
    elif config.pop=="tail":
        popularity="tail"
    else:
        raise Error("Popularity type error.")
    print(args.attack_eval_args)
    attack_eval_args = Bunch(args.attack_eval_args)
    attack_eval_args.target_items_path=attack_eval_args.target_items_path+popularity+"_"+config.tag+".npz"

    #attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+config.att_type+"_fake_data_best_"+config.tag+".npz"
    attack_eval_args.fake_data_path=None
    # Load fake data (and combine with normal training data) if path provided.
    n_fakes = 0

    # Evaluate victim model performance.
    for victim_args in attack_eval_args.victims:
        victim_args = Bunch(victim_args)
        victim_args.tag=config.tag
        trainer_class = victim_args.model["trainer_class"]
        trainer = trainer_class(n_users=n_users+n_fakes,
                                n_items=n_items,
                                args=victim_args)
        trainer.fit(train_data, test_data)
        # Load target items and evaluate attack performance.
        target_items = np.load(attack_eval_args.target_items_path)['target_items']
        trainer.validate(train_data, test_data, target_items)
        print("\n")
    

if __name__ == "__main__":
    args = importlib.import_module(config.config_file)
    
    args.seed=int(time.time())
    set_seed(args.seed, args.use_cuda)
    generate(args,config)
    
    config.config_file="evaluate_attack_args"
    evaluate_args=importlib.import_module(config.config_file)
    evaluate(evaluate_args,config)
