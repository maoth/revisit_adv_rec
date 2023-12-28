import argparse
import importlib
import sys
import numpy as np
import time
import copy

from bunch import Bunch
from sklearn.decomposition import NMF,PCA
from sklearn.cluster import KMeans

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
parser.add_argument("-alpha",type=float,default=1.0)
parser.add_argument("-transfer",type=int,default=0)
parser.add_argument("-cluster",type=int,default=0)
config = parser.parse_args()

def get_target_users(model,training_dataset,user_amounts,item_amounts,target_item,cluster):
    current_model=model
    recommendation_of_normal_users=model.recommend(training_dataset[:user_amounts],item_amounts)
    user_item_ranks=np.zeros_like(recommendation_of_normal_users)
    for i in range(user_amounts):
        for j in range(item_amounts):
            user_item_ranks[i,recommendation_of_normal_users[i,j]]=j
    rank_of_items=[np.where(row==target_item)[0] for row in recommendation_of_normal_users]
    rank_of_items=np.asarray(rank_of_items)
    rank_of_items=np.reshape(rank_of_items,(1,len(rank_of_items)))[0]
    real_target_users=[]
    for i in range(len(rank_of_items)):
        real_target_users.append((i,rank_of_items[i]))
    real_target_users=sorted(real_target_users,key=lambda x:x[1])
    real_target_users=real_target_users[-1000:] #target user amounts 
    real_target_users_id=[ele[0] for ele in real_target_users]
    real_target_users_rank=[ele[1] for ele in real_target_users]
    all_users=np.arange(n_users-attack_gen_args.n_fake_users)
    remain_users=np.setdiff1d(all_users,real_target_users_id)
    remain_users_ranks=np.setdiff1d(all_users,real_target_users_rank)
    if cluster==0:
        sample_users_id=np.random.choice(remain_users,size=1000,replace=False)
    else:
        cluster_cnt=len(remain_users)/1000
        pca=PCA(n_components=100)
        compressed_remain_users=pca.fit_transform(remain_users)
        kmeans = KMeans(n_clusters=cluster_cnt, random_state=0)
        kmeans.fit(compressed_remain_users)
        cluster_labels = kmeans.labels_
        cluster_avg=[]
        for i in range(cluster_cnt):
            cluster_data=remain_users_ranks[cluster_labels==i]
            avg=np.mean(cluster_data)
            cluster_avg.append(i,avg)
        cluster_avg=np.array(cluster_avg)
        if cluster==1:
            cluster_id=cluster_avg[np.argmax(cluster_avg[:,1])]
        if cluster==-1:
            cluster_id=cluster_avg[np.argmin(cluster_avg[:,1])]
        sample_users_id=remain_users[cluster_labels==cluster_id]                

    target_users_rank=np.mean(user_item_ranks[real_target_users_id],axis=0)
    sample_users_rank=np.mean(user_item_ranks[sample_users_id],axis=0)

    real_trigger_item_id=np.argmin(target_users_rank-sample_users_rank)

    return real_target_users_id,real_trigger_item_id

def generate(config,revise,target_items,target_users,trigger_item,n_users,n_items,train_data,test_data,attack_gen_args):

    # Load data.
    #print(config)
    print("dataset={}".format(args.data_path))

    print("attack method={}".format(args.attack_gen_args))
    print("Loading data from {}".format(args.data_path))
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    print("surrogate model={}".format(attack_gen_args.surrogate))



    print("n_users: {}, n_items: {}".format(n_users, n_items))
    print("popularity={}, fake user ratio={}, fake users={}, unroll epochs={}".format(attack_gen_args.target_item_popularity,attack_gen_args.n_fakes,attack_gen_args.n_fake_users, attack_gen_args.unroll_steps))    
    attack_gen_args.target_items = target_items
    attack_gen_args.trigger_item=None
    if config.att_type=="DQN":
        DQN_attack(n_users=n_users,n_items=n_items,train_data=train_data,test_data=test_data,args=attack_gen_args)
    else:
        train_target_users=copy.deepcopy(train_data)
        if revise>0:
            attack_gen_args.trigger_item=np.zeros_like(1)
            attack_gen_args.trigger_item=trigger_item
            train_target_users=train_target_users[target_users,:]

        adv_trainer_class = attack_gen_args.trainer_class
        adv_trainer = adv_trainer_class(n_users=n_users,n_items=n_items,args=attack_gen_args,attack_ver=revise)
        if train_target_users.shape[0]==n_users:
            train_target_users=np.zeros((1,n_items))
        adv_trainer.fit(train_data, test_data,train_target_users)

def evaluate(args,config,revise,trigger_item):
    args.data_path="./data/"+config.dataset
    print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    n_fake_users = int(n_users * config.ratio/100)   
    data_loader.n_users=n_users+n_fake_users
    for i in range(n_fake_users):
        data_loader.user2id[n_users+i]=n_users+i
    n_users, n_items=data_loader.n_users, data_loader.n_items
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
    #print(args.attack_eval_args)
    attack_eval_args = Bunch(args.attack_eval_args)
    attack_eval_args.target_items_path=attack_eval_args.target_items_path+popularity+"_"+config.tag+".npz"

    n_fakes = 0
    if revise==-1:
        attack_eval_args.fake_data_path=None
    else:        
        if config.att_type=='WRMF_SGD':
            attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+"Sur-WeightedMF-sgd"
        if config.att_type=='WRMF_ALS':
            attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+"Sur-WeightedMF-als"
        if config.att_type=='RandFilter':
            attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+"Random"
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
        target_items[0]=trigger_item
        result=trainer.validate(train_data, test_data, target_items)   
        trigger_item_HR=result["TargetHR@20"]
    return target_item_HR,trigger_item_HR    
    

if __name__ == "__main__":
    args = importlib.import_module(config.config_file)
    args.seed=int(time.time())
    set_seed(args.seed, args.use_cuda)
    args.data_path="./data/"+config.dataset
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

    attack_gen_args = Bunch(args.attack_gen_args)
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

    attack_gen_args.unroll_steps=config.unroll
    attack_gen_args.tag=config.tag
    if config.alpha>0:
        attack_gen_args.alpha=config.alpha
    if config.alpha<0:
        attack_gen_args.alpha=-1.0/config.alpha
    print(attack_gen_args.alpha)
    
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    attack_gen_args.n_fakes=config.ratio/100
    attack_gen_args.n_fake_users = int(n_users * attack_gen_args.n_fakes)
    org_train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()
    
    train_data_first_half=org_train_data[1::2,:]
    train_data_second_half=org_train_data[::2,:]
    if config.transfer==0:
        train_data=train_data_first_half
        n_users=train_data_first_half.get_shape()[0]

    '''
    HRsum0=0
    HRsum1=0
    HRsumt=0
    HRsumBenign=0
    HRsumtBenign=0
    cnt=0
    '''
    args.seed=int(time.time())
    set_seed(args.seed, args.use_cuda)
    target_items = sample_target_items(train_data,n_samples=attack_gen_args.n_target_items,popularity=attack_gen_args.target_item_popularity,use_fix=attack_gen_args.use_fixed_target_item,output_dir=attack_gen_args.output_dir,tag=attack_gen_args.tag)
    attack_gen_args.target_items = target_items
    model=init_model(attack_gen_args,n_users,n_items,train_data,test_data)
    revised_target_users,trigger_item=get_target_users(model,train_data,n_users,n_items,target_items[0],config.cluster)
    print("target item={},trigger item={}".format(target_items,trigger_item))
    '''
    config.config_file="evaluate_attack_args"
    evaluate_args=importlib.import_module(config.config_file)
    targetHR,triggerHR=evaluate(evaluate_args,config,-1,trigger_item)
    
    
    #HRsumBenign+=targetHRb*100
    #HRsumtBenign+=triggerHRb*100
    config.config_file="generate_attack_args"
    generate(config,0,target_items,revised_target_users,trigger_item,n_users,n_items,train_data,test_data,attack_gen_args)
    config.config_file="evaluate_attack_args"
    evaluate_args=importlib.import_module(config.config_file)
    targetHR0,triggerHR=evaluate(evaluate_args,config,0,trigger_item)
    #HRsum0+=targetHR0*100
    '''
    
    config.config_file="generate_attack_args"
    generate(config,1,target_items,revised_target_users,trigger_item,n_users,n_items,train_data,test_data,attack_gen_args)
    config.config_file="evaluate_attack_args"
    evaluate_args=importlib.import_module(config.config_file)
    targetHR,triggerHR=evaluate(evaluate_args,config,1,trigger_item)
    
    print("targetHR={:.4f} triggerHR={:.4f}".format(targetHR*100,triggerHR*100))
    #HRsum1+=targetHR1*100
    #HRsumt+=triggerHR*100
    '''
    print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(targetHRb*100,triggerHRb*100,targetHR0*100,targetHR1*100,triggerHR*100))    
        if triggerHRb<triggerHR:
            cnt=cnt+1
        if i%10==9:
            print("Round {}".format(i+1))
            print("benign HR={:.2f},benign trigger HR={:.2f} revisit HR={:.2f},target item HR={:.2f},trigger item HR={:.2f}".format(HRsumBenign/(i+1),HRsumtBenign/(i+1),HRsum0/(i+1),HRsum1/(i+1),HRsumt/(i+1)))
            print("trigger item HR increase:{}".format(cnt))
    '''