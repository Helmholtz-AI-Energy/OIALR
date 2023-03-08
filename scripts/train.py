import logging
import os

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.pretty import pprint

import madonna
from madonna import utils

try:
    config_name = os.environ["CONFIG_NAME"]
except KeyError:
    config_name = "train.yaml"

# A logger for this file
log = logging.getLogger(__name__)  # testing! possibly remove later


@hydra.main(config_path="../configs/", config_name=config_name, version_base=hydra.__version__)
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.slurm_id = os.environ["SLURM_JOBID"]
    # from madonna.training_pipeline import train
    # Applies optional utilities
    # utils.extras(config)
    pprint(config)
    #
    # optimizer = madonna.utils.get_optimizer(config, model)
    # sch, warmup = madonna.utils.get_lr_schedules(config, optimizer, 10)
    log.info("comm init")
    rank, size = utils.comm.init_and_set_config_rank_size(config)
    with open_dict(config):
        config["world_size"] = size
        config["rank"] = rank
    log.info(f"Rank: {rank}, world size: {size}")

    if config.training.trainer == "slime":
        fn = madonna.trainers.slime_trainer.main
    elif config.training.trainer == "image-baseline":
        fn = madonna.trainers.images.baseline.main
    else:
        raise ValueError(f"unknonw trainer: {config.trainer.trainer}")

    if rank == 0:
        _ = utils.tracking.setup_mlflow(config, verbose=False)
        pprint(config)

        # run_id -> adaptive needs to be unique, roll random int?
        # run_name = f"" f"full-rank-everybatch-{os.environ['SLURM_JOBID']}"
        with mlflow.start_run() as run:
            mlflow.log_param("Slurm jobid", os.environ["SLURM_JOBID"])
            run_name = f"{config['name']}-" + run.info.run_name
            mlflow.set_tag("mlflow.runName", run_name)
            # print("run_name:", run_name)
            # print("tracking uri:", mlflow.get_tracking_uri())
            # print("artifact uri:", mlflow.get_artifact_uri())
            log.info(f"run_name: {run_name}")
            log.info(f"tracking uri: {mlflow.get_tracking_uri()}")
            log.info(f"artifact uri: {mlflow.get_artifact_uri()}")
            madonna.utils.tracking.log_config(config)
            # hydra.utils.call(config.training.script, config)
            fn(config)
            # exit(0)
    else:
        # hydra.utils.call(config.training.script, config)
        fn(config)


if __name__ == "__main__":
    main()
