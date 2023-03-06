from __future__ import annotations

import os
import subprocess
import time

import mlflow
from mlflow.entities import Experiment
from omegaconf import DictConfig

from .utils import only_on_rank_n

__all__ = ["setup_mlflow"]


only_on_rank_n(run_on_rank=0)


def setup_mlflow(config: dict, verbose: bool = False) -> Experiment | None:  # noqa: E302
    """Setup MLFlow server, connect to it, and set the experiment

    Parameters
    ----------
    config: dict
        Config dictionary
    verbose: bool
        if this should print the mlflow server on rank0
    rank: int, optional
        process rank

    exp_id: int
        MLFlow experiment ID
    """
    print("in SETUP_MLFLOW: TESTING TO MAKRE SURE THIS ONLY RUNS ON RANK 0! REMOVE WHEN CONFIRMED")
    # if rank is not None:
    #     if dist.is_initialized() and dist.get_rank() != 0:
    #         return
    # if rank != 0:
    #     return
    restart_mlflow_server(config)
    # Connect to the MLFlow client for tracking this training - only on rank0!
    mlflow_server = f"http://127.0.0.1:{config['mlflow']['port']}"
    mlflow.set_tracking_uri(mlflow_server)
    if verbose:
        print(f"MLFlow connected to server {mlflow_server}")

    experiment = mlflow.set_experiment(f"{config['dataset']}-{config['arch']}")
    return experiment


only_on_rank_n(run_on_rank=0)


def restart_mlflow_server(config: dict):  # noqa: E302
    """Kill any existing mlflow servers and restart them

    Parameters
    ----------
    config: dict
        Config dictionary
    """

    # kill mlflow and start the server
    processname = "gunicorn"
    tmp = os.popen("ps -Af").read()
    proccount = tmp.count(processname)
    if proccount == 0:
        subprocess.Popen(["pkill", "-f", "gunicorn"])
        print("Starting MLFlow server")
        mlflow_server_cmd = [
            "/usr/local/bin/mlflow",
            "server",
            "--backend-store-uri",
            f"{config['mlflow']['tracking_uri']}",
            "--default-artifact-root",
            f"{config['mlflow']['artifact_location']}",
            "--port",
            f"{config['mlflow']['port']}",
        ]
        print("mlflow cmd", mlflow_server_cmd)
        _ = subprocess.Popen(mlflow_server_cmd)
        time.sleep(2)


only_on_rank_n(0)


def log_config(config, keys=None):  # noqa: E302
    # mlflow.log_params(config)
    for k in config:
        if isinstance(config[k], DictConfig):
            for k2 in config[k]:
                if isinstance(config[k][k2], DictConfig):
                    log_config(config[k][k2], keys=f"{k}-{k2}")
                else:
                    # mlflow.log_param(f"{cat}-{k}-{k2}", config[cat][k][k2])
                    if keys is not None:
                        lpkeys = f"{keys}-{k}-{k2}"
                    else:
                        lpkeys = f"{k}-{k2}"
                    mlflow.log_param(lpkeys, config[k][k2])
        else:
            if keys is not None:
                lpkeys = f"{keys}-{k}"
            else:
                lpkeys = f"{k}"
            mlflow.log_param(lpkeys, config[k])
