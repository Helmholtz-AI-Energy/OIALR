from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import mlflow
from mlflow.entities import Experiment
from omegaconf import DictConfig

from .utils import only_on_rank_n

uc2 = socket.gethostname().startswith("uc2")
horeka = socket.gethostname().startswith("hkn")

__all__ = ["setup_mlflow"]


@only_on_rank_n(run_on_rank=0)
def setup_mlflow(config: DictConfig, verbose: bool = False) -> Experiment | None:  # noqa: E302
    """Setup MLFlow server, connect to it, and set the experiment

    Parameters
    ----------
    config: dict
        Config dictionary
    verbose: bool
        if this should print the mlflow server on rank0
    """
    # if rank is not None:
    #     if dist.is_initialized() and dist.get_rank() != 0:
    #         return
    # if rank != 0:
    #     return
    create_mlflow_backup(config)
    restart_mlflow_server(config)
    # Connect to the MLFlow client for tracking this training - only on rank0!
    mlflow_server = f"http://127.0.0.1:{config.tracking.mlflow.port}"
    mlflow.set_tracking_uri(mlflow_server)
    if verbose:
        print(f"MLFlow connected to server {mlflow_server}")

    experiment = mlflow.set_experiment(f"{config.data.dataset}-{config.model.name}")
    return experiment


@only_on_rank_n(run_on_rank=0)
def create_mlflow_backup(config: DictConfig):
    if uc2:
        tracking = Path(str(config.tracking.mlflow.tracking_uri_uc2).split(":")[-1])
    elif horeka:
        tracking = Path(str(config.tracking.mlflow.tracking_uri_horeka).split(":")[-1])
    else:
        tracking = Path(str(config.tracking.mlflow.tracking_uri).split(":")[-1])
    folder = tracking.parent
    backup_folder = folder / "backup"
    backup_folder.mkdir(exist_ok=True)
    shutil.copy(tracking, backup_folder / tracking.name)


@only_on_rank_n(run_on_rank=0)
def restart_mlflow_server(config: DictConfig):  # noqa: E302
    """Kill any existing mlflow servers and restart them

    Parameters
    ----------
    config: DictConfig
        Config dictionary
    """

    # kill mlflow and start the server
    processname = "gunicorn"
    tmp = os.popen("ps -Af").read()
    proccount = tmp.count(processname)
    if proccount == 0:
        # get location of MLFlow
        loc = subprocess.run(["/usr/bin/which", "mlflow"], stdout=subprocess.PIPE)
        mlflow_cmd = loc.stdout.decode("utf-8")[:-1]  # slice off \n

        subprocess.Popen(["pkill", "-f", "gunicorn"])
        print("Starting MLFlow server")

        if uc2:
            tracking = config.tracking.mlflow.tracking_uri_uc2
            artifact = config.tracking.mlflow.artifact_location_uc2
        elif horeka:
            tracking = config.tracking.mlflow.tracking_uri_horeka
            artifact = config.tracking.mlflow.artifact_location_horeka
        else:
            tracking = config.tracking.mlflow.tracking_uri
            artifact = config.tracking.mlflow.artifact_location

        mlflow_server_cmd = [
            f"{mlflow_cmd}",
            "server",
            "--backend-store-uri",
            f"{tracking}",
            "--default-artifact-root",
            f"{artifact}",
            "--port",
            f"{config.tracking.mlflow.port}",
        ]
        print("mlflow cmd", mlflow_server_cmd)
        _ = subprocess.Popen(mlflow_server_cmd)
        time.sleep(2)


@only_on_rank_n(0)
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
        elif k == "_partial_":
            continue
        else:
            if keys is not None:
                lpkeys = f"{keys}-{k}"
            else:
                lpkeys = f"{k}"
            mlflow.log_param(lpkeys, config[k])
