import os
from unittest import mock
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# ## set the tracking URI
mlflow.set_tracking_uri('/home/intellect/Uday/mlflow_test/mlruns')
# ## set the registry uri
mlflow.set_registry_uri('/home/intellect/Uday/mlflow_test/mlruns')

## set the experiement name. If experiment not exist, it creates new experiment.
mlflow.set_experiment('basic_dl')


if __name__ == "__main__":
    mlflow.start_run()
    # mlflow.set_tag('mlflow.source.type', 'PROJECT')
    # mlflow.set_tag('mlflow.source.name', 'https://github.com/UdiBhaskar/mlflow-basic-tracking/blob/exp-0/dl_basic_log.py')
    mlflow.set_tag("mlflow.user", 'Uday')
    logging_params = dict()
    logging_params['model'] = 'roberta-base'
    logging_params['learning_rate'] = 2.5e-5
    logging_params['epochs'] = 20
    logging_params['batch_size'] = 16
    logging_params['seed_val'] = 24
    logging_params['optimizer'] = 'adam'
    mlflow.log_params(logging_params)

    model = AutoModel.from_pretrained('roberta-base')
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)

    config_dict = model.config.to_dict()

    mlflow.log_dict(config_dict, 'model/config.json')

    logging_metrics_epoch = dict()

    random_f1 = np.random.rand(110)
    random_f1.sort()
    f1_socres = random_f1[random_f1<0.95][-20:]
    loss_vals = random_f1[random_f1<0.3][-20:][::-1]

    for i in range(logging_params['epochs']):
        logging_metrics_epoch = dict()
        logging_metrics_epoch['f1'] = f1_socres[i]
        logging_metrics_epoch['loss'] = loss_vals[i]
        mlflow.log_metrics(logging_metrics_epoch, step=i)
    mlflow.pytorch.log_model(model, "ptmodel")

    state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i,
                "loss": logging_metrics_epoch['loss'],
            }
    mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")
    mlflow.log_artifact('/home/intellect/Uday/mlflow_test/test_scripts/log_details/dl_basic_log.py', 'code')
    mlflow.set_tag('version', '2')
    mlflow.set_tag("source.git.repoURL", "https://github.com/UdiBhaskar/mlflow-basic-tracking.git")
    mlflow.set_tag('source.git.branch', 'exp-0')
    mlflow.end_run()
