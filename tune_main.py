import os
import ray
import pandas as pd
from data import *
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from pathlib import Path
import ray.train as train
from ray.train import Checkpoint, CheckpointConfig, DataConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchCheckpoint, TorchTrainer
import tempfile
import torch
import json
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import RobertaModel,RobertaTokenizer
from ray.train.torch import get_device
from ray.tune.logger.wandb import WandbLoggerCallback
import config
import wandb
from dotenv import load_dotenv
from model import *
from process_data import *
ray.data.DatasetContext.get_current().execution_options.preserve_order = True

num_workers = 1
resources_per_worker={"CPU": 1, "GPU": 1}

load_dotenv()

wandb_api_key = "70e95a405ea4aec8d0a637460407bf21c69436f4"

num_classes = 3
train_loop_config = {
    "dropout_p": 0.5,
    "lr": 1e-4,
    "lr_factor": 0.8,
    "lr_patience": 3,
    "num_epochs": 10,
    "batch_size": 256,
    "num_classes": num_classes,
}

options = ray.data.ExecutionOptions(preserve_order=True)
dataset_config = DataConfig(
    datasets_to_split=["train"],
    execution_options=options)
scaling_config = ScalingConfig(
    num_workers=num_workers,
    use_gpu=bool(resources_per_worker["GPU"]),
    resources_per_worker=resources_per_worker
)
checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")

wandb_callback = WandbLoggerCallback(
        project=config.WANDB_PROJECT,
        api_key=wandb_api_key,
        upload_checkpoints=True,
        log_config=True,
    )
run_config = RunConfig(
    callbacks=[wandb_callback],
    checkpoint_config=checkpoint_config,
    # storage_path=config.LOCAL_DIR,
    # local_dir=config.LOCAL_DIR
)

train_ds,val_ds, preprocessor = get_train_data("train.csv")
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    datasets={"train": train_ds, "val": val_ds},
    dataset_config=dataset_config,
    metadata={"class_to_index": preprocessor.class_to_index}
)

initial_params = [{"train_loop_config": {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}}]
search_alg = HyperOptSearch(points_to_evaluate=initial_params)
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2) 

param_space = {
    "train_loop_config": {
        "dropout_p": tune.uniform(0.3, 0.9),
        "lr": tune.loguniform(1e-5, 5e-4),
        "lr_factor": tune.uniform(0.1, 0.9),
        "lr_patience": tune.uniform(1, 10),
    },
    # "log_level":"DEBUG"

}

scheduler = AsyncHyperBandScheduler(
    max_t=train_loop_config["num_epochs"],  # max epoch (<time_attr>) per trial
    grace_period=5,  # min epoch (<time_attr>) per trial
)

NUM_RUNS = 1

tune_config = tune.TuneConfig(
    metric="val_loss",
    mode="min",
    search_alg=search_alg,
    scheduler=scheduler,
    num_samples=NUM_RUNS,
)

tuner = Tuner(
    trainable=trainer,
    run_config=run_config,
    param_space=param_space,
    tune_config=tune_config,
)

results = tuner.fit()