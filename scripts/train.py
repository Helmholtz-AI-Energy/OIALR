import logging
import os

import hydra
import mlflow
import torch
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.pretty import pprint

import madonna
import wandb
from madonna import utils

try:
    config_name = os.environ["CONFIG_NAME"]
except KeyError:
    config_name = "train.yaml"

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name=config_name, version_base=hydra.__version__)
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        try:
            config.slurm_id = os.environ["SLURM_JOBID"]
        except KeyError:
            pass
    # from madonna.training_pipeline import train
    # Applies optional utilities
    # utils.extras(config)
    # pprint(config)
    #
    # optimizer = madonna.utils.get_optimizer(config, model)
    # sch, warmup = madonna.utils.get_lr_schedules(config, optimizer, 10)
    pprint("before comm init")
    rank, size = utils.comm.init_and_set_config_rank_size(config)
    with open_dict(config):
        config["world_size"] = size
        config["rank"] = rank
        config["global_batch_size"] = config.data.local_batch_size * size

    if config.training.trainer == "slime":
        fn = madonna.trainers.slime_trainer.main
    elif config.training.trainer == "image-baseline":
        fn = madonna.trainers.images.baseline.main
    elif config.training.trainer == "MyOpt":
        fn = madonna.trainers.my_opt_trainer.main
    elif config.training.trainer == "ortho_fix_train":
        fn = madonna.trainers.ortho_fix_train.main
    elif config.training.trainer == "ortho_sam_train":
        fn = madonna.trainers.ortho_sam_trainer.main
    else:
        raise ValueError(f"unknown trainer: {config.trainer.trainer}")

    if config.tracking.sweep is not None:
        if rank == 0:

            def func():
                return fn(config)

            wandb.agent(config.tracking.sweep, function=func, count=1)
        else:
            fn(config)
    elif rank == 0 and config.enable_tracking and config.tracking == "mlflow":
        _ = utils.tracking.setup_mlflow(config, verbose=False)

        run_id = None
        skip_namechange = False
        if config.training.checkpoint is not None:
            checkpoint = torch.load(config.training.checkpoint)
            if "mlflow_run_name" in checkpoint and config.training.resume_run:
                skip_namechange = True
                run_id = str(checkpoint["mlflow_run_name"])
                print(run_id)

        # run_id -> adaptive needs to be unique, roll random int?
        # run_name = f"" f"full-rank-everybatch-{os.environ['SLURM_JOBID']}"
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_param("Slurm jobid", os.environ["SLURM_JOBID"])
            if not skip_namechange:
                # dont need to change the name of existing runs
                run_name = f"{config['name']}-" + run.info.run_name
                mlflow.set_tag("mlflow.runName", run_name)

            # print("run_name:", run_name)
            # print("tracking uri:", mlflow.get_tracking_uri())
            # print("artifact uri:", mlflow.get_artifact_uri())
            log.info(f"Rank: {rank}, world size: {size}")

            log.info(f"run_name: {run_name}")
            log.info(f"tracking uri: {mlflow.get_tracking_uri()}")
            log.info(f"artifact uri: {mlflow.get_artifact_uri()}")
            madonna.utils.tracking.log_config(config)
            # hydra.utils.call(config.training.script, config)
            fn(config)
    elif rank == 0 and config.enable_tracking and config.tracker == "wandb":
        # wandb.login()
        # wandb.init(
        #     name=str(config.name),
        #     project=str(config.tracking.project),
        #     config=OmegaConf.to_container(config),
        #     group=f"{config.data.dataset}-{config.model.name}",
        # )
        fn(config)
        # wandb.finish()
    else:
        # hydra.utils.call(config.training.script, config)
        fn(config)


if __name__ == "__main__":
    main()
