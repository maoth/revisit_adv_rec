import argparse
import importlib
import sys
import numpy as np
import time
import copy

from bunch import Bunch

from data.data_loader import DataLoader
from utils.utils import set_seed, sample_target_items,stack_csrdata, load_fake_data
from framework import DQN_attack,init_model
from adversary_example_generator_v2 import generate_target_users     

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="generate_attack_args")
parser.add_argument("-dataset", type=str)
parser.add_argument("-att_type", type=str)
parser.add_argument("-ratio", type=float)
parser.add_argument("-pop",type=str)
parser.add_argument("-unroll",type=int)
parser.add_argument("-tag",type=str)
#parser.add_argument("-revise",type=int,default=0)
config = parser.parse_args()


def generate(args,config,revise):
    # Load data.
    #print(config)
    args.data_path="./data/"+config.dataset
    #print("dataset={}".format(args.data_path))
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
    #print("attack method={}".format(args.attack_gen_args))
    #print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    #print("n_users: {}, n_items: {}".format(n_users, n_items))

    # Train & evaluate adversarial users.
    attack_gen_args = Bunch(args.attack_gen_args)
    #print("surrogate model={}".format(attack_gen_args.surrogate))
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
    data_loader.n_users=n_users+attack_gen_args.n_fake_users
    for i in range(attack_gen_args.n_fake_users):
        data_loader.user2id[n_users+i]=n_users+i
    n_users, n_items=data_loader.n_users, data_loader.n_items
    #print("n_users: {}, n_items: {}".format(n_users, n_items))
    train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()
    #print("popularity={}, fake user ratio={}, fake users={}, unroll epochs={}".format(attack_gen_args.target_item_popularity,attack_gen_args.n_fakes,attack_gen_args.n_fake_users, attack_gen_args.unroll_steps))
    target_items = sample_target_items(
        train_data,
        n_samples=attack_gen_args.n_target_items,
        popularity=attack_gen_args.target_item_popularity,
        use_fix=attack_gen_args.use_fixed_target_item,
        output_dir=attack_gen_args.output_dir,
        tag=attack_gen_args.tag
    )
    attack_gen_args.target_items = target_items
    if config.att_type=="DQN":
        DQN_attack(n_users=n_users,n_items=n_items,train_data=train_data,test_data=test_data,args=attack_gen_args)
    else:
        if revise>0:
            if revise>1:
                target_items=np.append(target_items,[n_items-1])
                attack_gen_args.target_items=target_items
            for i in range(attack_gen_args.n_fake_users):
                if revise>1:
                    train_data[n_users-i-1,n_items-1]=1
                train_data[n_users-i-1,target_items[0]]=1
            model=init_model(attack_gen_args,n_users,n_items,train_data,test_data)
            revised_target_users=generate_target_users(model,train_data,n_users,n_items,target_items[0])
            new_train_data=copy.deepcopy(train_data)
            new_train_data=new_train_data[revised_target_users,:]
            train_data=new_train_data
            n_users=len(revised_target_users)
        adv_trainer_class = attack_gen_args.trainer_class
        adv_trainer = adv_trainer_class(n_users=n_users,
                                        n_items=n_items,
                                        args=attack_gen_args)
        adv_trainer.fit(train_data, test_data)

def evaluate(args,config,revise):
    args.data_path="./data/"+config.dataset
    #print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    #print("n_users: {}, n_items: {}".format(n_users, n_items))
    n_fake_users = int(n_users * config.ratio/100)   
    data_loader.n_items=n_items+1
    data_loader.item2id[n_items]=n_items
    data_loader.n_users=n_users+n_fake_users
    for i in range(n_fake_users):
        data_loader.user2id[n_users+i]=n_users+i
    n_users, n_items=data_loader.n_users, data_loader.n_items
    #print("n_users: {}, n_items: {}".format(n_users, n_items))    
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
    #print(args.attack_eval_args)
    attack_eval_args = Bunch(args.attack_eval_args)
    attack_eval_args.target_items_path=attack_eval_args.target_items_path+popularity+"_"+config.tag+".npz"

    n_fakes = 0
    if revise==-1:
        attack_eval_args.fake_data_path=None
    else:        
        if config.att_type=='WRMF_SGD':
            attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+"Sur-WeightedMF-sgd"
        attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+"_fake_data_best_"+config.tag+".npz"
    # Load fake data (and combine with normal training data) if path provided.

    if attack_eval_args.fake_data_path:
        fake_data = load_fake_data(attack_eval_args.fake_data_path)
        train_data = stack_csrdata(train_data, fake_data)
        n_fakes = fake_data.shape[0]
        '''
        print("Statistics of fake data: "
              "n_fakes={}, avg_clicks={:.2f}".format(
                n_fakes, fake_data.sum(1).mean()))
        '''

    # Evaluate victim model performance.
    # evaluate output HR20
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
        result=trainer.validate(train_data, test_data, target_items)
        target_item_HR=result["TargetHR@20"]
        target_items[0]=n_items-1
        trainer.validate(train_data, test_data, target_items)   
        trigger_item_HR=result["TargetHR@20"]
    return target_item_HR,trigger_item_HR    
    

if __name__ == "__main__":
    args = importlib.import_module(config.config_file)
    
    #args.seed=int(time.time())
    set_seed(args.seed, args.use_cuda)
    HRsum0=0
    HRsum1=0
    HRsum2=0
    HRsumBenign=0
    for i in range(100):
        print(i)
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHRb,triggerHR=evaluate(evaluate_args,config,-1)
        HRsumBenign+=targetHRb*100
        config.config_file="generate_attack_args"
        generate(args,config,0)
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHR0,triggerHR=evaluate(evaluate_args,config,0)
        HRsum0+=targetHR0*100
        config.config_file="generate_attack_args"
        generate(args,config,1)
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHR1,triggerHR=evaluate(evaluate_args,config,1)
        HRsum1+=targetHR1*100
        config.config_file="generate_attack_args"
        generate(args,config,2)
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHR2,triggerHR=evaluate(evaluate_args,config,2)
        HRsum2+=targetHR2*100
        print(targetHRb*100,targetHR0*100,targetHR1*100,targetHR2*100)
        if i%10==0:
            print("Round {}".format(i+1))
            print("benign HR={:.2f},revisit HR={:.2f},target user HR={:.2f},trigger HR={:.2f}".format(HRsumBenign/(i+1),HRsum0/(i+1),HRsum1/(i+1),HRsum2/(i+1)))