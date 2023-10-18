from functools import partial

from metrics.ranking_metrics import *
from trainers import *
from trainers.losses import *

data_path = "./data/gowalla"  # # Dataset path and loader
use_cuda = True  # If using GPU or CPU
seed = 1  # Random seed
metrics = [PrecisionRecall(k=[50]), NormalizedDCG(k=[50])]

shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./outputs/",
}

""" Victim model hyper-parameters."""
vict_itemcf = {
    **shared_params,
    "epochs": 1,
    "lr": 0.0,
    "l2": 0.0,
    "save_feq": 1,
    "batch_size": 128,
    "valid_batch_size": 128,
    "model": {
        "trainer_class": ItemCFTrainer,
        "model_name": "ItemCF",
        "knn": 50
    }
}
vict_cml = {
    **shared_params,
    "epochs": 100,
    "lr": 1e-3,
    "l2": 1e-4,
    "save_feq": 100,
    "batch_size": 512,
    "valid_batch_size": 32,
    "model": {
        "trainer_class": NCFTrainer,
        "model_name": "CML",
        "num_factors": 256
    }
}
vict_ncf = {
    **shared_params,
    "epochs": 90,
    "lr": 3e-4,
    "l2": 1e-4,
    "save_feq": 90,
    "batch_size": 512,
    "valid_batch_size": 32,
    "model": {
        "trainer_class": NCFTrainer,
        "model_name": "NeuralCF",
        "hidden_dims": [128],
        "num_factors": 256,
        "ac": "tanh",
    }
}
vict_wmf_sgd = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-2,
    "l2": 1e-5,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "WeightedMF-sgd",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}
vict_user_ae = {
    **shared_params,
    "epochs": 70,
    "lr": 1e-3,
    "l2": 1e-6,
    "save_feq": 70,
    "batch_size": 1024,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": UserAETrainer,
        "model_name": "UserVAE",
        "model_type": "UserVAE",
        "hidden_dims": [512, 256],
        "betas": [0.0, 1e-4, 1.0],
        "recon_loss": mult_ce_loss
    }
}
vict_item_ae = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-3,
    "l2": 1e-6,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": ItemAETrainer,
        "model_name": "ItemAE",
        "hidden_dims": [256, 128],
        "recon_loss": partial(mse_loss, weight=20)
    }
}

""" Attack evaluation hyper-parameters."""
attack_eval_args = {
    **shared_params,

    # Path to the fake data.
    # If None, then evaluate clean performance without attack.
    #"fake_data_path": "./outputs/",
    "fake_data_path": None,
    # Path to the target items.
    "target_items_path": "./outputs/sampled_target_items_1_",

    # List of victim models to evaluate.
    "victims": [vict_wmf_sgd]
}
#"victims": [vict_itemcf, vict_cml, vict_ncf, vict_user_ae, vict_wmf_sgd, vict_item_ae]