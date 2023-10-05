from pathlib import Path

from omegaconf import DictConfig, OmegaConf, errors, open_dict


def obj_train():
    # print("obj fn")
    import os

    rank = os.environ["RANK"]

    base_config_path = Path("/hkfs/work/workspace/scratch/qv2382-madonna/madonna/configs/")
    base_config = OmegaConf.load(base_config_path / "ortho_train.yaml")
    iteration = 0
    if (base_config_path / f"propulate-{rank}.yaml").exists():
        rank_conf = OmegaConf.load(base_config_path / f"propulate-{rank}.yaml")
        iteration = rank_conf.iteration
    with open_dict(base_config):
        base_config["iteration"] = iteration + 1
        base_config["handle"] = rank
        base_config.hydra.job_logging.handlers.file.filename = (
            "${hydra.runtime.output_dir}/${hydra.job.name}-rank${handle}-${iteration}.log"
        )
    # return
    # print(base_config)
    OmegaConf.save(base_config, base_config_path / f"propulate-{rank}.yaml")

    import json

    import hydra

    import madonna

    @hydra.main(config_path="../configs", config_name=f"propulate-{rank}.yaml", version_base=hydra.__version__)
    def train(config: DictConfig):
        # need to load the other config and update it
        overrides = OmegaConf.load(
            f"/hkfs/work/workspace/scratch/qv2382-madonna/madonna/configs/tmp/newest_config_{rank}.yaml",
        )
        for param in overrides:
            # try:
            #     OmegaConf.update(config, param, overrides[param])
            # except errors.ConfigAttributeError:
            with open_dict(config):
                config[param] = overrides[param]

        with open_dict(config):
            config[
                "loaded_file"
            ] = f"/hkfs/work/workspace/scratch/qv2382-madonna/madonna/configs/tmp/newest_config_{rank}.yaml"
        # Imports can be nested inside @hydra.main to optimize tab completion
        # https://github.com/facebookresearch/hydra/issues/934
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            try:
                config.slurm_id = os.environ["SLURM_JOBID"]
            except KeyError:
                pass
        with open_dict(config):
            config["world_size"] = 1
            config["rank"] = 0
            config["global_batch_size"] = config.data.local_batch_size

        fn = madonna.trainers.ortho_fix_train.main
        try:
            fn(config)
        except ValueError:
            ret_dict = {"train_loss": 2.5, "train_top1": 0.0, "val_loss": 2.5, "val_top1": 0.0}

            # propulate minimizes...
            ret_dict["train_top1"] = 1 - (ret_dict["train_top1"] * 0.01)
            ret_dict["val_top1"] = 1 - (ret_dict["val_top1"] * 0.01)
            # out_file_root = Path("/hkfs/work/workspace/scratch/qv2382-madonna/madonna/configs/tmp/")
            out_file = Path("/hkfs/work/workspace/scratch/qv2382-madonna/madonna/configs/tmp/")
            with open(out_file / f"{rank}-output.txt", "w") as convert_file:
                # convert_file.write(json.dumps(ret_dict))
                json.dump(ret_dict, convert_file)

    train()
