import json
import yaml

import pandas as pd
import numpy as np
from urllib.parse import urlparse
import mlflow
import torch
from transformers import AutoModel


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# ## set the tracking URI
mlflow.set_tracking_uri('http://0.0.0.0:5000/')

with open("mlflow_project.yml", 'r') as stream:
    project_config = yaml.safe_load(stream)

## set the experiement name. If experiment not exist, it creates new experiment.
mlflow.set_experiment('basic_dl')


if __name__ == "__main__":

    mlflow.start_run() ##starts the runtime
    logging_params = dict()

    #training parameters
    training_params = json.load(open("training_config.json"))
    mlflow.log_params(training_params)  ##logs the parameters

    model = AutoModel.from_pretrained('roberta-base')
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)

    config_dict = model.config.to_dict()

    mlflow.log_dict(config_dict, 'model/config.json')

    logging_metrics_epoch = dict()

    random_f1 = np.random.rand(110)
    random_f1.sort()
    f1_socres = random_f1[random_f1<0.7][-20:]
    loss_vals = random_f1[random_f1<0.5][-20:][::-1]

    for i in range(training_params['epochs']):
        # ---------------train the model here ---------------------
        logging_metrics_epoch = dict()
        logging_metrics_epoch['f1'] = f1_socres[i]
        logging_metrics_epoch['loss'] = loss_vals[i]
        mlflow.log_metrics(logging_metrics_epoch, step=i)

    ##save the model
    mlflow.pytorch.log_model(model, "ptmodel")

    ## if i want to save the modelweights, optimizer state for training from the specific state.
    state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i,
                "loss": logging_metrics_epoch['loss'],
            }
    mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")

    ##We can also log the code we executed. or any file ( json, text, .code, datafiles)
    mlflow.log_artifact('train.py', 'code')

    ##log all project config as tags.
    mlflow.set_tags(project_config)

    ##end the run
    mlflow.end_run()
