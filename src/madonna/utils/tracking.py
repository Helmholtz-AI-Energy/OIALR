from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf, listconfig

import wandb
from wandb import finish, init

from .utils import only_on_rank_n

WANDB_ARTIFACT_PREFIX = "wandb-artifact://"
uc2 = socket.gethostname().startswith("uc2")
horeka = socket.gethostname().startswith("hkn")

# __all__ = ["setup_mlflow"]
log = logging.getLogger(__name__)


@only_on_rank_n(run_on_rank=0)
def setup_mlflow(config: DictConfig, verbose: bool = False):  # noqa: E302
    """Setup MLFlow server, connect to it, and set the experiment

    Parameters
    ----------
    config: dict
        Config dictionary
    verbose: bool
        if this should print the mlflow server on rank0
    """
    import mlflow

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
    import mlflow

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
def log_config(config, wandb_logger, keys=None):  # noqa: E302
    # mlflow.log_params(config)
    for k in config:
        if isinstance(config[k], DictConfig):
            for k2 in config[k]:
                if isinstance(config[k][k2], DictConfig):
                    log_config(config[k][k2], wandb_logger, keys=f"{k}-{k2}")
                else:
                    # mlflow.log_param(f"{cat}-{k}-{k2}", config[cat][k][k2])
                    if keys is not None:
                        lpkeys = f"{keys}-{k}-{k2}"
                    else:
                        lpkeys = f"{k}-{k2}"
                    wandb_logger.log({lpkeys: config[k][k2]})
        elif k == "_partial_":
            continue
        else:
            if keys is not None:
                lpkeys = f"{keys}-{k}"
            else:
                lpkeys = f"{k}"
            wandb_logger.log({lpkeys: config[k][k2]})


@only_on_rank_n(0)
def log_config_wandb(config, keys=None):  # noqa: E302
    # mlflow.log_params(config)
    for k in config:
        if isinstance(config[k], DictConfig):
            for k2 in config[k]:
                if isinstance(config[k][k2], DictConfig):
                    wandb.log(config[k][k2], keys=f"{k}-{k2}")
                else:
                    # mlflow.log_param(f"{cat}-{k}-{k2}", config[cat][k][k2])
                    if keys is not None:
                        lpkeys = f"{keys}-{k}-{k2}"
                    else:
                        lpkeys = f"{k}-{k2}"
                    wandb.log(lpkeys, config[k][k2])
        elif k == "_partial_":
            continue
        else:
            if keys is not None:
                lpkeys = f"{keys}-{k}"
            else:
                lpkeys = f"{k}"
            wandb.log(lpkeys, config[k])


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix) :]


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    model_artifact_name = "run_" + run_id + "_model"
    return run_id, project, model_artifact_name


def check_wandb_resume(config):
    if config.training.resume:
        if config.training.checkpoint.startswith(WANDB_ARTIFACT_PREFIX):
            if config.global_rank not in [-1, 0]:  # For resuming DDP runs
                run_id, project, model_artifact_name = get_run_info(config.training.checkpoint)
                api = wandb.Api()
                artifact = api.artifact(project + "/" + model_artifact_name + ":latest")
                modeldir = artifact.download()
                config.weights = str(Path(modeldir) / "last.pt")
            return True
    return None


class WandbLogger:
    def __init__(self, config, job_type="Training"):
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, wandb.run
        self.config = config

        primative = OmegaConf.to_container(config, resolve=True)
        tags = []
        if "tags" in config.tracking:
            tag = config.tracking.tags
            if isinstance(tag, (list, listconfig.ListConfig)):
                tags.extend(tag)
            else:
                tags.append(tag)
        if config.baseline:
            tags.append("baseline")
        if isinstance(config.training.checkpoint, str):  # checks resume from artifact
            if config.training.checkpoint.startswith(WANDB_ARTIFACT_PREFIX):
                run_id, project, model_artifact_name = get_run_info(config.training.checkpoint)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, "install wandb to resume wandb runs"
                # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
                self.wandb_run = wandb.init(
                    id=run_id,
                    project=project,
                    resume="allow",
                    group=f"{config.data.dataset}-{config.model.name}",
                    tags=tags,
                )
                config = self.wandb_run.config
                config.training.checkpoint = model_artifact_name
        elif self.wandb and config.tracking.sweep is not None:
            self.wandb_run = wandb.init()
        elif self.wandb:
            # init for new run
            self.wandb_run = wandb.init(
                config=primative,
                resume="allow",
                project=config.tracking.project,
                name=config.name,
                job_type=job_type,
                tags=tags,
                group=f"{config.data.dataset}-{config.model.name}",
            )
        log_file = config["log_file_out"]
        self.wandb_run.save(log_file)
        self.setup_training(config)

    def setup_training(self, config):
        self.log_dict, self.current_epoch = {}, 0  # Logging Constants

        if isinstance(config.training.resume, str):
            modeldir, _ = self.download_model_artifact(config)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                OmegaConf.update(config, self.wandb_run.config, merge=True)

    def download_model_artifact(self, config):
        if config.training.checkpoint.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(
                remove_prefix(config.training.checkpoint, WANDB_ARTIFACT_PREFIX) + ":latest",
            )
            assert model_artifact is not None, "Error: W&B model artifact doesn't exist"
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get("epochs_trained")
            total_epochs = model_artifact.metadata.get("total_epochs")
            assert epochs_trained < total_epochs, f"training to {total_epochs} epochs is finished, nothing to resume."
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, config, epoch, fitness_score, best_model=False):
        model_artifact = wandb.Artifact(
            "run_" + wandb.run.id + "_model",
            type="model",
            metadata={
                "original_url": str(path),
                "epochs_trained": epoch + 1,
                "project": config.tracking.project,
                "total_epochs": config.training.epochs,
                "fitness_score": fitness_score,
            },
        )
        model_artifact.add_file(str(path / "last.pt"), name="last.pt")
        wandb.log_artifact(
            model_artifact,
            aliases=["latest", "epoch " + str(self.current_epoch), "best" if best_model else ""],
        )
        print("Saving model artifact on epoch ", epoch + 1)

    def log(self, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        if self.wandb_run:
            wandb.log(self.log_dict)
            self.log_dict = {}
            self.current_epoch += 1

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                wandb.log(self.log_dict)
            # upload the output log from hydra (hopefully that doesnt fuck up...)
            self.wandb.run.finish()
