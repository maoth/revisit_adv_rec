import argparse
import importlib

from bunch import Bunch
import numpy as np

from data.data_loader import DataLoader
from utils.utils import set_seed, stack_csrdata, load_fake_data

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="evaluate_attack_args")
parser.add_argument("-dataset", type=str)
parser.add_argument("-att_type", type=str)
parser.add_argument("-tag", type=str)
parser.add_argument("-pop", type=str)
config = parser.parse_args()


def main(args,config):
    # Load data.
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

    attack_eval_args = Bunch(args.attack_eval_args)
    attack_eval_args.target_items_path=attack_eval_args.target_items_path+popularity+"_"+config.tag+".npz"

    #attack_eval_args.fake_data_path=attack_eval_args.fake_data_path+config.att_type+"_fake_data_best_"+config.tag+".npz"
    attack_eval_args.fake_data_path=None
    # Load fake data (and combine with normal training data) if path provided.
    n_fakes = 0
    if attack_eval_args.fake_data_path:
        fake_data = load_fake_data(attack_eval_args.fake_data_path)
        train_data = stack_csrdata(train_data, fake_data)
        n_fakes = fake_data.shape[0]
        print("Statistics of fake data: "
              "n_fakes={}, avg_clicks={:.2f}".format(
                n_fakes, fake_data.sum(1).mean()))

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

    set_seed(args.seed, args.use_cuda)
    main(args,config)
