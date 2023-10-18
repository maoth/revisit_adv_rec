import argparse
import importlib
import sys

from bunch import Bunch

from data.data_loader import DataLoader
from utils.utils import set_seed, sample_target_items

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="generate_attack_args")
parser.add_argument("-dataset", type=str)
parser.add_argument("-att_type", type=str)
parser.add_argument("-ratio", type=float)
parser.add_argument("-pop",type=str)
parser.add_argument("-unroll",type=int)
parser.add_argument("-tag",type=str)
config = parser.parse_args()


def main(args,config):
    # Load data.

    args.data_path="./data/"+config.dataset
    print("dataset={}".format(args.data_path))
    if config.att_type=="WRMF_ALS":
        args.attack_gen_args=args.attack_gen_args_wrmf_als 
    elif config.att_type=="WRMF_SGD":
        args.attack_gen_args=args.attack_gen_args_wrmf_sgd
    elif config.att_type=="RandFilter":
        args.attack_gen_args=args.attack_gen_args_randfilter
    else:
        raise Error("Attack method error.")
    print("attack method={}".format(args.attack_gen_args))
    print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()

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
    print("popularity={},fake user ratio={},unroll epochs={}".format(attack_gen_args.target_item_popularity,attack_gen_args.n_fakes,attack_gen_args.unroll_steps))
    target_items = sample_target_items(
        train_data,
        n_samples=attack_gen_args.n_target_items,
        popularity=attack_gen_args.target_item_popularity,
        use_fix=attack_gen_args.use_fixed_target_item,
        output_dir=attack_gen_args.output_dir,
        tag=attack_gen_args.tag
    )
    attack_gen_args.target_items = target_items
    #print(attack_gen_args)

    adv_trainer_class = attack_gen_args.trainer_class
    adv_trainer = adv_trainer_class(n_users=n_users,
                                    n_items=n_items,
                                    args=attack_gen_args)
    adv_trainer.fit(train_data, test_data)


if __name__ == "__main__":
    args = importlib.import_module(config.config_file)
    
    set_seed(args.seed, args.use_cuda)
    main(args,config)
