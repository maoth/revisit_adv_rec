from functools import partial

from metrics.ranking_metrics import *
from trainers import *
from trainers.losses import *

data_path = "./data/ml-1m"  # # Dataset path and loader
use_cuda = True  # If using GPU or CPU
seed = 1  # Random seed
metrics = [PrecisionRecall(k=[50]), NormalizedDCG(k=[50])]

shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./outputs/",
}

""" Surrogate model hyper-parameters."""
sur_item_ae = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-3,
    "l2": 1e-6,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": ItemAETrainer,
        "model_name": "Sur-ItemAE",
        "hidden_dims": [256, 128],
        "recon_loss": partial(mse_loss, weight=20)
    }
}
sur_wmf_sgd = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-2,
    "l2": 1e-5,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "Sur-WeightedMF-sgd",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}
sur_wmf_als = {
    **shared_params,
    "epochs": 10,
    "lr": 1e-3,
    "l2": 5e-2,
    "save_feq": 10,
    "batch_size": 1,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "Sur-WeightedMF-als",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "als"
    }
}

sur_wmf_dqn = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-3,
    "l2": 5e-2,
    "save_feq": 10,
    "batch_size": 16,
    "valid_batch_size": 16,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "Sur-WeightedMF-sgd",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}

""" Attack generation hyper-parameters for all methods, tuned with grid-search."""
#original trainer_class adversarial, using to test randfilter
attack_gen_args_item_ae_pd = {
    **shared_params,
    "trainer_class": BlackBoxAdvTrainer,
    "attack_type": "random",
    "n_target_items": 1,
    "target_item_popularity": "head",
    "use_fixed_target_item": False,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 0,

    "adv_lr": 1.0,
    "adv_momentum": 0.95,

    "proj_threshold": 0.05,
    "click_targets": True,

    # Args for surrogate model.
    "surrogate": sur_item_ae
}

attack_gen_args_item_ae = {
    **shared_params,
    "trainer_class": BlackBoxAdvTrainer,
    "attack_type": "adversarial",
    "n_target_items": 5,
    "target_item_popularity": "head",
    "use_fixed_target_item": True,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 5,

    "adv_lr": 1.0,
    "adv_momentum": 0.95,

    "proj_threshold": 0.1,
    "click_targets": False,

    # Args for surrogate model.
    "surrogate": sur_item_ae
}

#original unroll_steps=10, occurs CUDA out of memory
#original use_fixed=True
attack_gen_args_wmf_sgd = {
    **shared_params,
    "trainer_class": BlackBoxAdvTrainer,
    "attack_type": "adversarial",
    "n_target_items": 1,
    "target_item_popularity": "head",
    "use_fixed_target_item": False,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 10,

    "adv_lr": 1.0,
    "adv_momentum": 0.95,

    "proj_threshold": 0.05,
    "click_targets": False,

    # Args for surrogate model.
    "surrogate": sur_wmf_sgd
}

attack_gen_args_wmf_als = {
    **shared_params,
    "trainer_class": BlackBoxAdvTrainer,
    "attack_type": "adversarial",
    "n_target_items": 1,
    "target_item_popularity": "head",
    "use_fixed_target_item": False,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 0,

    "adv_lr": 1.0,
    "adv_momentum": 0.95,

    "proj_threshold": 0.1,
    "click_targets": True,

    # Args for surrogate model.
    "surrogate": sur_wmf_als
}
# Using the best attacking method (wmf_sgd). Note this method will requires ~25GB RAM
# since it needs more unroll steps.

attack_gen_args_dqn_sgd = {
    **shared_params,
    "trainer_class": BlackBoxAdvTrainer,
    "attack_type": "adversarial",
    "n_target_items": 1,
    "target_item_popularity": "head",
    "use_fixed_target_item": False,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 3,
    "unroll_steps": 0,

    "adv_lr": 1.0,
    "adv_momentum": 0.95,

    "proj_threshold": 0.1,
    "click_targets": True,
    "clusters": 100,

    # Args for surrogate model.
    "surrogate": sur_wmf_dqn
}

attack_gen_args_wrmf_sgd = attack_gen_args_wmf_sgd
attack_gen_args_wrmf_als = attack_gen_args_wmf_als
attack_gen_args_randfilter = attack_gen_args_item_ae_pd
attack_gen_args_dqn=attack_gen_args_dqn_sgd
attack_gen_args = attack_gen_args_item_ae_pd