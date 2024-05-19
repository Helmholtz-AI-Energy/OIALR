import json
import logging
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import reload
from pathlib import Path
from shutil import copy

from omegaconf import DictConfig, OmegaConf, errors, open_dict

# from multiprocessing import Process


def objective(search_params):
    # import hydra
    # del hydra
    # gc.collect()
    # # hydra.__dict__.clear()
    # import hydra
    # hydra.core.global_hydra.GlobalHydra.hydra = None
    import hydra
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs/", job_name="propulate", version_base=hydra.__version__)
    config = hydra.compose(config_name="ortho_train")
    for param in search_params:
        try:
            OmegaConf.update(config, param, search_params[param])
        except errors.ConfigAttributeError:
            with open_dict(config):
                config[param] = search_params[param]
    with open_dict(config):
        config.name = "propulate-svd"
    # OmegaConf.update(config, "data.data_dir_horeka", f"{os.environ['TMPDIR']}/cifar10-{rank}/CIFAR10")
    # tags the runs to enable keeping propulate runs together
    tags = ["propulate"]
    if "tags" in config.tracking:
        tag = config.tracking.tags
        if isinstance(tag, list):
            tags.extend(tag)
        else:
            tags.append(tag)
    with open_dict(config):
        config.tracking.tags = tags
    # TODO: get rank!!
    out_file_root = Path("/hkfs/work/workspace/scratch/CHANGE/ME-madonna2/madonna/configs/tmp/")
    out_file_root.mkdir(exist_ok=True)
    out_file = out_file_root / f"newest_config_{rank}.yaml"
    print(f"saving out file {out_file}")
    with open(out_file, "w") as f:
        OmegaConf.save(config, f)

    # import hydra
    # hydra.__dict__.clear()
    # del sys.modules["hydra"]
    # gc.collect()
    # import hydra
    # print(hydra.__dict__)
    # hydra = reload(hydra)

    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # subprocess.call(['python', '-c', 'import propulate_train; propulate_train.obj_train()'])

    output_file = out_file_root / f"{rank}-output.txt"
    try:
        os.remove(str(output_file))
    except FileNotFoundError:
        pass
    print(output_file)

    p = subprocess.Popen(
        [
            # "echo", "here;",
            "python",
            "-c",
            "print('h'); import os;"
            + f"os.environ['RANK']=str({rank});"
            + "from scripts import prop_train;"
            + "prop_train.obj_train()",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=os.environ,
        shell=False,
    )
    # subprocess.call
    while not output_file.exists():
        output, errs = p.communicate()
        # print(output, errs)
        time.sleep(1)
    # p.wait()
    # output, errs = p.poll()
    print(output[-800:])
    if errs != "":
        print(errs)
    p.kill()
    # train()
    with open(output_file, "r") as outputs:
        ret_dict = json.loads(outputs.read())
    print(ret_dict)
    return ret_dict["val_top1"]


def stage_data(path="/hkfs/home/dataset/datasets/CIFAR10/"):
    path = Path(path)
    if rank % 4 != 0:
        return  # f"{os.environ['TMP']}/CIFAR10"
    dest = Path(f"{os.environ['TMP']}/cifar10/CIFAR10")
    print(f"Creating directory on tmp: {dest}")
    # test = Path("/hkfs/home/dataset/datasets/CIFAR10/train/cifar-10-batches-py/data_batch_1")
    # print(test.exists())
    dest.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    def copy_file(src_path, dest_dir):
        # copy source file to dest file
        _ = copy(src_path, dest_dir)

    with ThreadPoolExecutor(max_workers=70) as executor:
        futures = []
        for f in path.rglob("*"):
            if not f.is_file():
                continue
            futures.append(executor.submit(copy_file, f, dest))
            # print('staging', f)
        for fut in futures:
            _ = fut.result()
    print(f"Staging time: {time.perf_counter() - start_time}")


if __name__ == "__main__":
    import propulate
    from mpi4py import MPI
    from propulate.utils import get_default_propagator

    rank = MPI.COMM_WORLD.Get_rank()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4)
    # stage_data()
    search_space = {
        "training.lr": (0.00005, 0.001),
        "training.max_grad_norm": (-1.0, 10.0),
        # Optimizer things
        "training.optimizer.weight_decay": (0.00001, 0.5),
        # "training.optimizer.beta1": (0.01, 0.999),
        # "training.optimizer.beta2": (0.01, 0.999),
        # Scheduler stuff
        # "training.lr_schedule.epochs": (1, 300),
        "training.lr_schedule.min_lr": (0.000005, 0.001),
        "training.lr_schedule.warmup_lr": (0.000005, 0.0002),
        "training.lr_schedule.lr_k_decay": (0.1, 2.0),
        # "training.lr_schedule.lr_cycle_mul": (0.75, 3),
        "training.lr_schedule.warmup_epochs": (1, 20),  # TODO: should this be up to 200?
        # "training.lr_schedule.lr_cycle_decay": (0.1, 1.0),
        # SVD stuff
        # "training.fixing_method.keep_last_layer": (0, 1),
        # "training.fixing_method.keep_first_layer": (0, 1),
        "training.fixing_method.sigma_cutoff_fraction": (0.01, 0.9),
        "training.fixing_method.stability_frequency": (200, 1000),
        "training.fixing_method.delay": (1000, 10000),
        "training.fixing_method.stability_update_delay": (0, 5),
    }
    num_generations = 200000
    pop_size = 2 * MPI.COMM_WORLD.size
    # GPUS_PER_NODE = 4
    rng = random.Random(MPI.COMM_WORLD.rank)
    islands = MPI.COMM_WORLD.size // 4

    def uniquify(path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + str(counter) + extension
            counter += 1
        return path

    log_path = Path("/hkfs/work/workspace/scratch/CHANGE/ME-madonna2/madonna/logs/propulate/cifar10-1/")
    log_path.mkdir(exist_ok=True, parents=True)
    propulate.set_logger_config(
        level=logging.INFO,
        log_file=uniquify(log_path / "propulate.log"),
        log_to_stdout=True,
        colors=True,
    )
    propagator = get_default_propagator(pop_size, search_space, 0.7, 0.4, 0.1, rng=rng)
    islands = propulate.Islands(
        objective,
        propagator,
        rng,
        generations=num_generations,
        num_islands=islands,
        migration_probability=0.9,
        checkpoint_path=Path("/hkfs/work/workspace/scratch/CHANGE/ME-madonna2/hpsearch/cifar10-1/"),
    )
    # TODO: set this up to change propulate's logging debug level!!!
    islands.evolve(top_n=1, logging_interval=1, debug=1)
