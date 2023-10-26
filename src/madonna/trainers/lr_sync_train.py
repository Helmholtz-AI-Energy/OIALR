from __future__ import annotations

import logging
import os
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from enum import Enum
from pathlib import Path

import hydra

# from mpi4py import MPI
import numpy as np

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
import omegaconf
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
from omegaconf import OmegaConf, open_dict
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.pretty import pprint
from torch.autograd.grad_mode import no_grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torchmetrics import MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import madonna
import wandb

best_acc1 = 0
log = logging.getLogger(__name__)


def main(config):  # noqa: C901
    log.info(config)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(hydra_cfg["runtime"]["output_dir"])
    # log.info("here is some log testing")
    # log.info("more stuff")
    # log.info("do i need more things?")
    for handler in log.parent.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
    with open_dict(config):
        config["log_file_out"] = log_file

    # return {"train loss": 0.75, "train top1": 52, "val loss": 0.54, "val top1": 85}
    if config.tracker == "wandb":
        wandb_run = madonna.utils.tracking.check_wandb_resume(config)
    if config.training.resume and not wandb_run:  # resume an interrupted run
        ckpt = config.training.checkpoint
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
    if config.training.checkpoint_out_root is not None:
        with omegaconf.open_dict(config):
            config.save_dir = madonna.utils.utils.increment_path(
                Path(config.training.checkpoint_out_root) / config.name,
            )  # increment run
        if config.rank == 0:
            log.info(f"save_dir: {config.save_dir}")

        save_dir = Path(config.save_dir)

        # Directories
        wdir = save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        log.info(f"out_dir (wdir): {wdir}")
        last = wdir / "last.pt"
    else:
        wdir = None
    # results_file = save_dir / 'results.txt'
    # f = open(results_file, 'w')
    wandb_logger = madonna.utils.tracking.WandbLogger(config) if config.rank == 0 and config.enable_tracking else None

    # ----------------- Sweep stuffs --------------------------
    if config.tracking.sweep is not None:
        from mpi4py import MPI

        if config.rank == 0:
            sweep_config = wandb.config._as_dict()
            for key in sweep_config:
                if key == "_wandb":
                    continue
                # print(key, key in config)
                # if key in config:
                OmegaConf.update(config, key, sweep_config[key])
            # dict_config.update(sweep_config)
            print(sweep_config)
            # need to send and update all the dicts
        dict_config = OmegaConf.to_container(config, resolve=False)

        comm = MPI.COMM_WORLD
        new_dict = comm.bcast(dict_config, root=0)

        dict_config["rank"] = dist.get_rank()
        config = OmegaConf.create(new_dict)
        with open_dict(config):
            config.rank = dist.get_rank() if dist.is_initialized() else 0

        if config.rank == 0:
            pprint(dict(config))
            # madonna.utils.tracking.log_config(config, wandb_logger)
            if wandb_logger is not None:
                wandb_logger.log(dict_config)
            # wandb_logger.wandb_run.config.update(dict_config)
    # -------- Random Seed init --------------------------
    if config.seed is None:
        seed = torch.seed()
        random.seed(torch.seed())
    else:
        seed = int(config.seed)

        cudnn.benchmark = True
        cudnn.deterministic = True
        # torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=5)
        # if dist.is_initialized():
        #     scale = dist.get_rank() % 4
        # else:
        #     scale = 0
        random.seed(seed)
        torch.manual_seed(seed)

    # if config.rank == 0:
    log.info(f"Seed: {torch.initial_seed()}")
    # -------- End Random Seed init --------------------------

    if config.cpu_training:
        gpu = None
        log.debug("NOT using GPUs!")
        device = torch.device("cpu")
    else:
        if dist.is_initialized():
            gpu = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
            log.debug(f"Using GPU: {gpu}")
        else:
            log.info(f"available GPUS: {torch.cuda.device_count()}")
            gpu = 0
            # log.info(f"available GPUS: {torch.cuda.device_count()}")
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    # print('before get model')
    model = madonna.utils.get_model(config)
    if not config.cpu_training:
        model.cuda(gpu)

    if config.training.sync_batchnorm and dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # # TODO: fix model repr
    # if dist.get_rank() == 0:
    #     print(model.model)
    if config.baseline:
        model = DDP(model)  # , device_ids=[config.rank])
        log.info("using DDP baseline model")
        # if dist.get_rank() == 0:
        #     print(model)
    # elif dist.is_initialized():
    else:
        model_hold = hydra.utils.instantiate(config.training.fixing_method)
        model = model_hold(model).to(device)
        log.info("using SVD model")
    # else:
    #     log.info("using baseline 1 process model")
    # print(model)

    # model_param_dict = madonna.lrsync.sync.get_param_dict_with_svds(model)

    criterion = madonna.utils.get_criterion(config)
    optimizer = madonna.utils.get_optimizer(config, model, lr=config.training.lr)

    # set optimizer for references for SVD model to reset shapes of state params
    if not config.baseline:
        model.set_optimizer(optimizer)

    dset_dict = madonna.utils.datasets.get_dataset(config)
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    # sam_opt = madonna.optimizers.svd_sam.SAM(params=params, base_optimizer=optimizer, rho=0.05, adaptive=False)
    train_len = len(train_loader)
    scheduler, _ = madonna.utils.get_lr_schedules(config, optimizer, train_len)

    # optionally resume from a checkpoint
    # Reminder: when resuming from a single checkpoint, make sure to call init_model with
    start_epoch = config.training.start_epoch
    if config.training.checkpoint is not None:  # TODO: FIXME!!
        if os.path.isfile(config.training.checkpoint):
            print(f"=> loading checkpoint: {config.training.checkpoint}")
            if torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{gpu}"
                checkpoint = torch.load(config.training.checkpoint, map_location=loc)
            start_epoch = checkpoint["epoch"]

            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config.training.checkpoint,
                    start_epoch,
                ),
            )
            # if not config.baseline:
            #     optimizer.param_groups[-1]["lr"] = config.training.sigma_optimizer.min_lr
            # # optimizer should have the correct LR from the loading point
            # if not config.baseline:
            #     if "next_stability_iteration" in checkpoint:
            #         model.next_stability_iteration = checkpoint["next_stability_iteration"]
            #     if "call_count" in checkpoint:
            #         model.call_count = checkpoint["call_count"]
            if scheduler is not None and start_epoch > 0:
                if True:  # args.sched_on_updates: FIXME
                    scheduler.step_update(start_epoch * len(train_loader))
                else:
                    scheduler.step(start_epoch)
        else:
            print(f"=> no checkpoint found at: {config.training.checkpoint}")

    out_base = None
    if "checkpoint_out_root" in config.training and config.training.checkpoint_out_root is not None:
        out_base = Path(config.training.checkpoint_out_root)
        model_name = config.model.name
        dataset_name = config.data.dataset
        out_base /= "baseline" if config.baseline else "svd"
        out_base = out_base / f"{model_name}-{dataset_name}" / f"bs-{config.data.local_batch_size * config.world_size}"
        out_base.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving model to: {out_base}")

    # # if config['evaluate']:
    # #     validate(val_loader, dlrt_trainer, config)
    # #     return

    train_metrics = MetricCollection(
        [
            MulticlassF1Score(num_classes=config.data.classes),
            MulticlassPrecision(num_classes=config.data.classes),
            MulticlassRecall(num_classes=config.data.classes),
        ],
    ).to(device)
    val_metrics = MetricCollection(
        [
            MulticlassF1Score(num_classes=config.data.classes),
            MulticlassPrecision(num_classes=config.data.classes),
            MulticlassRecall(num_classes=config.data.classes),
        ],
    ).to(device)

    # ============= make first layer the same =================
    # if config.training.init_method is not None:

    # =========================================================
    # print(model)

    scaler = torch.cuda.amp.GradScaler(enabled=config.model.autocast)
    rank = dist.get_rank() if dist.is_initialized() else 0

    refactory_warmup = None
    len_train = len(train_loader)
    best_fitness = 0.0
    last_val_top1s = []
    # TODO: flag for full_rank training
    model_param_dict = None
    # if not config.baseline and model.local_low_rank_model is not None and dist.is_initialized():
    #     log.info("partially syncing models from low rank")
    #     model_param_dict = madonna.lrsync.sync.get_param_dict_with_svds(model)
    #     madonna.lrsync.sync.sync_topn_singulars_oialr(
    #         model_param_dict, vecs_to_send=config.training.sync.vecs, method=config.training.sync.method
    #     )
    # elif not config.baseline and dist.is_initialized():
    #     log.info("partially syncing models in full rank")
    #     madonna.lrsync.sync.sync_model_in_low_rank(model, config)
    #     # madonna.lrsync.sync.sync_topn_singulars_oialr(model_param_dict, topn=10)
    if not config.baseline and model.local_low_rank_model is not None:
        model.blur_svd_layers()

    just_synced = False

    for epoch in range(start_epoch, config.training["epochs"]):
        torch.cuda.reset_peak_memory_stats()
        if config["rank"] == 0:
            # console.rule(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            lr_list, prnt_str = get_lrs(optimizer)
            log.info(f"Begin epoch {epoch} LRs: {prnt_str}")
            lrs = {"lr": lr_list[0]}
            if config.enable_tracking:
                # wandb_logger.log(metrics, step=epoch * len_train)
                wandb_logger.log(lrs)
        if dist.is_initialized() and config.data.distributed_sample and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if just_synced:
            model.blur_svd_layers()
            just_synced = False

        train_loss, last_loss, refactory_warmup, train_t1 = train(
            train_loader=train_loader,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            epoch=epoch,
            device=device,
            config=config,
            scaler=scaler,
            lr_scheduler=scheduler,
            refactory_warmup=refactory_warmup,
            mixup=dset_dict["mixup"],
            wandb_logger=wandb_logger,
            metrics=train_metrics,
        )

        # # -------------- pruning ----------------------------
        # if not config.baseline and epoch >= config.training.prune_epoch:
        #     # fc -> resnets ;;; vit -> head
        #     madonna.lrsync.prune.prune_model(model, threshold=1e-3, perc=None, last_layer_name="fc")
        # # -------------- end pruning ----------------------------
        if (
            not config.baseline
            and model_param_dict is None
            and model.local_low_rank_model is not None
            and dist.is_initialized()
        ):
            model_param_dict = madonna.lrsync.sync.get_param_dict_with_svds(model)
        if (
            not config.baseline
            and epoch >= config.training.sync.first_epoch
            and epoch % config.training.sync.epoch_freq == 0
            and dist.is_initialized()
        ):
            if model_param_dict is not None:
                madonna.lrsync.sync.sync_low_rank_model(
                    model_param_dict,
                    vecs_to_send=config.training.sync.vecs,
                    method=config.training.sync.method,
                    sort=config.training.sort,
                )
                pass
                # madonna.models.utils.reset_opt_state(optimizer=optimizer)
                # # self.reset_all_states(self.optimizer)
                # # self.optimizer.reset_shapes_of_sigma(self)
                # model.optimizer.zero_grad(set_to_none=True)
            else:
                madonna.lrsync.sync.sync_full_rank_model_in_low_rank(model, config)
                pass
            just_synced = True
            # model.blur_svd_layers()
            # madonna.models.utils.reset_opt_state(optimizer=optimizer)

        if config.rank == 0:
            log.info(f"Average Training loss across process space: {train_loss}")
            par, zpar = 0, 0
            for p in model.parameters():
                par += p.numel()
                zpar += (p.abs() < 1e-6).sum()
            log.info(f"Network Sparsity: {zpar / par}, zeros: {zpar} nonzero: {par}")
        # evaluate on validation set
        val_top1, val_loss = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            config=config,
            epoch=epoch,
            device=device,
            print_on_rank=config.rank == 0,
            pg=None,
            len_train=len_train,
            wandb_logger=wandb_logger,
            metrics=val_metrics,
        )

        if config.rank == 0:
            log.info(
                f"Average val loss across process space: {val_loss} " f"-> diff: {train_loss - val_loss}",
            )

        # log the percentage of params in use
        if config.rank == 0 and not config.baseline:
            if hasattr(model, "get_perc_params_all_layers"):
                perc, trainable, normal, compression_perc = model.get_perc_params_all_layers()
                if config.enable_tracking:
                    # mlflow.log_metric("perc_parmas", perc, step=epoch)
                    wandb_logger.log({"perc_parmas": perc})  # , step=(epoch + 1) * len_train)
                log.info(
                    f"% params: {perc:.3f}% trainable: {trainable} full: {normal} compression: {compression_perc:.3f}%",
                )
                # model.track_interior_slices_mlflow(config, epoch)

        # log metrics
        # train_metrics_epoch = train_metrics.compute()
        val_metrics_epoch = val_metrics.compute()
        log_dict = {
            # "train/f1": train_metrics_epoch["MulticlassF1Score"],
            # "train/precision": train_metrics_epoch["MulticlassPrecision"],
            # "train/recall": train_metrics_epoch["MulticlassRecall"],
            # "train/loss": train_loss,
            "val/f1": val_metrics_epoch["MulticlassF1Score"],
            "val/precision": val_metrics_epoch["MulticlassPrecision"],
            "val/recall": val_metrics_epoch["MulticlassRecall"],
            # "val/loss": val_loss,
        }

        # Save model
        if dist.is_initialized():
            wait = dist.barrier(async_op=True)
        if rank == 0 and config.enable_tracking:
            last_val_top1s.append(val_top1.item() if isinstance(val_top1, torch.Tensor) else val_top1)
            if len(last_val_top1s) > 10:
                slope, _ = np.polyfit(x=np.arange(10), y=np.array(last_val_top1s[-10:]), deg=1)
                log.info(f"Slope of Top1 for last 10 epochs: {slope:.5f}")

            if val_top1 > best_fitness:
                best_fitness = val_top1

            # print(train_metrics_epoch)
            log.info(
                f"Epoch end metrics: \n\t"
                # f"train f1/prec/rec: {log_dict['train/f1']:.4f} / "
                # f"{log_dict['train/precision']:.4f} / {log_dict['train/recall']:.4f}"
                f"val f1/prec/rec: {log_dict['val/f1']:.4f} / "
                f"{log_dict['val/precision']:.4f} / {log_dict['val/recall']:.4f}",
            )
            wandb_logger.log(log_dict)

            wandb_logger.end_epoch(best_result=best_fitness == val_top1)

            ckpt = {
                "epoch": epoch,
                "best_fitness": best_fitness,
                # 'training_results': results_file.read_text(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "wandb_id": wandb_logger.wandb_run.id if wandb_logger.wandb else None,
            }
            print("After ckpt")
            # Save last, best and delete
            if wdir is not None:
                torch.save(ckpt, last)
                # if best_fitness == val_top1:
                #     torch.save(ckpt, best)
                #     print("After 1st save")
                if best_fitness == val_top1:
                    torch.save(ckpt, wdir / "best_{:03d}.pt".format(epoch))
                    print("After best save")

                if epoch == 0:  # first
                    torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                    print("After 1st save")
                elif (
                    config.training.save_period != -1 and ((epoch + 1) % config.training.save_period) == 0
                ):  # on command
                    torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                    print("After periodic save")
                # elif epoch >= (config.training.epochs - 5):
                #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))

                # if wandb_logger.wandb and config.enable_tracking:
                #     if (
                #         (epoch + 1) % config.training.save_period == 0 and not epoch == config.training.epochs - 1
                #     ) and config.training.save_period != -1:
                #         wandb_logger.log_model(
                #             last.parent,
                #             config,
                #             epoch,
                #             val_top1,
                #             best_model=best_fitness == val_top1,
                #         )
                #         print("After wandb log model")
                del ckpt
        if dist.is_initialized():
            # wait here for the saving and such...it didnt work to have it afterwards
            wait.wait(timeout=timedelta(seconds=60))
        # # early stopping for imagenet...
        # if (val_top1 < 15. and epoch >= 5) or \
        #    (val_top1 < 60. and epoch >= 25) or \
        #    (val_top1 < 70. and epoch >= 50):  # or \
        #     #    (val_top1 < 75. and epoch >= 70):  # or \
        #     # (val_top1 < 78. and epoch >= 100):
        #     if rank == 0:
        #         log.info("Early stopping")
        #     break
        # set up next epoch after saving everything
        # wandb.log({"epoch": epoch}, step=(epoch + 1) * len_train)
        scheduler.step(epoch + 1, metric=val_loss)
        train_metrics.reset()
        val_metrics.reset()
    if rank == 0:
        log.info("End of run")
        wandb_logger.finish_run()
    # import json
    # val_top1 = val_top1 if not isinstance(val_top1, torch.Tensor) else val_top1.item()
    # ret_dict = {"train_loss": train_loss, "train_top1": train_t1, "val_loss": val_loss, "val_top1": val_top1}
    # # propulate minimizes...
    # ret_dict["train_top1"] = 1 - (ret_dict["train_top1"] * 0.01)
    # ret_dict["val_top1"] = 1 - (ret_dict["val_top1"] * 0.01)
    # print("from train", ret_dict)
    # # out_file_root = Path("/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/configs/tmp/")
    # out_file = Path("/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/configs/tmp/")
    # with open(out_file / f"{os.environ['RANK']}-output.txt", "w") as convert_file:
    #     # convert_file.write(json.dumps(ret_dict))
    #     json.dump(ret_dict, convert_file)
    # return ret_dict


def get_lrs(opt):
    out_lrs = []
    prnt_str = ""
    for group in opt.param_groups:
        out_lrs.append(group["lr"])
        prnt_str += f"group {len(out_lrs)}: lr {group['lr']:.6f}\t"
    return out_lrs, prnt_str


def train(
    train_loader,
    optimizer: madonna.optimizers.MixedSVDOpt,
    model,
    criterion,
    epoch,
    device,
    config,
    wandb_logger,
    metrics,
    lr_scheduler=None,
    scaler=None,
    log=log,
    refactory_warmup=None,
    mixup=None,
):
    train_time0 = time.perf_counter()
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )
    # switch to train mode
    model.train()
    if refactory_warmup is None:
        refactory_warmup = {}

    end = time.time()
    updates_per_epoch = len(train_loader)
    num_updates = epoch * updates_per_epoch
    for i, data in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        if hasattr(config.data, "dali") and config.data.dali:
            images = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        else:
            images = data[0]
            target = data[1]
        if mixup is not None:
            images, target = mixup(images, target)
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # if config.rank == 0 and i % 4 == 0:
        #     _, lrs = optimizer.get_lrs()
        #     print(f"LRS: {lrs}")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.model.autocast):
            # NOTE: some things dont play well with autocast - one should not put anything aside from the model in here
            output = model(images)
            loss = criterion(output, target)

        if torch.isnan(loss):
            for n, p in model.named_parameters():
                print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
            raise ValueError("NaN loss")
        scaler.scale(loss).backward()

        if config.training.max_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(output, target, topk=(1, 5), mixup=mixup is not None)
        losses.update(loss.item(), images.size(0))
        try:
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        except IndexError:
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
        # metrics.update(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        num_updates += 1
        # if steps_remaining > 0:
        #     for c, group in enumerate(optimizer.param_groups):
        #         group["lr"] += step_factors[c]
        #     # steps_remaining = lr_reset_steps - 1
        # else:
        #     # TODO: this will only work for timm schedulers
        lr_scheduler.step_update(num_updates=num_updates, metric=loss)

        # if argmax.std() == 0:
        #     log.error(f"ISSUE WITH NETWORK printing debugging info")
        #     for n, p in model.named_parameters():
        #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        #     raise ValueError
        argmax = torch.argmax(output, dim=1).to(torch.float32)

        if (i % config.training.print_freq == 0 or i == len(train_loader) - 1) and config["rank"] == 0:
            argmax = torch.argmax(output, dim=1).to(torch.float32)
            log.info(
                f"Argmax outputs s "
                f"mean: {argmax.mean().item():.5f}, max: {argmax.max().item():.5f}, "
                f"min: {argmax.min().item():.5f}, std: {argmax.std().item():.5f}",
            )
            progress.display(i + 1, log=log)

        optimizer.zero_grad()

    if config.rank == 0:
        log.info(f"Data Loading Time avg: {data_time.avg}")

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
    if config["rank"] == 0 and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        wandb_logger.log(
            {
                "train/loss": ls,
                "train/top1": t1,
                "train/top5": t5,
                "train/time": time.perf_counter() - train_time0,
            },
        )
    return losses.avg, loss, refactory_warmup, t1


@torch.no_grad()
def save_selected_weights(network, epoch, config):
    if not config.baseline:
        raise RuntimeError(
            "Weights should only be saved when running baseline! (remove for other data gathering)",
        )
    if not config.save_weights:
        return
    save_list = config.save_layers
    save_location = Path(config.save_parent_folder) / config.model.name / "baseline_weights"
    rank = dist.get_rank()

    if rank != 0:
        dist.barrier()
        return
    # save location: resnet18/name/epoch/[u, s, vh, weights
    for n, p in network.named_parameters():
        if n in save_list:
            print(f"saving: {n}")
            n_save_loc = save_location / n / str(epoch)
            n_save_loc.mkdir(exist_ok=True, parents=True)

            torch.save(p.data, n_save_loc / "p.pt")
    # print("finished saving")
    dist.barrier()


@torch.no_grad()
def validate(val_loader, model, criterion, config, epoch, device, print_on_rank, pg, len_train, wandb_logger, metrics):
    def run_validate(loader, base_progress=0):
        # rank = 0 if not dist.is_initialized() else dist.get_rank()
        with torch.no_grad():
            end = time.time()
            num_elem = len(loader) - 1
            for i, data in enumerate(loader):
                if hasattr(config.data, "dali") and config.data.dali:
                    images = data[0]["data"]
                    target = data[0]["label"].squeeze(-1).long()
                else:
                    images = data[0]
                    target = data[1]
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                data_time.update(time.time() - end)

                if torch.any(torch.isnan(images)):
                    # for n, p in model.named_parameters():
                    # print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                    raise ValueError("NaN in images... - VAL")

                # compute output
                output = model(images)
                loss = criterion(output, target)
                # argmax = torch.argmax(output.output, dim=1).to(torch.float32)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                metrics.update(output, target)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if torch.isnan(loss):
                    for n, p in model.named_parameters():
                        print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                    raise ValueError("NaN loss - VAL")
                # argmax = torch.argmax(output, dim=1).to(torch.float32)
                # # print(
                # #     f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, ",
                # #     f"min: {argmax.min().item()}, std: {argmax.std().item()}",
                # # )
                # # progress.display(i + 1, log=log)

                # if argmax.std() == 0:
                #     log.error(f"ISSUE WITH NETWORK printing debugging info")
                #     for n, p in model.named_parameters():
                #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                #     raise ValueError

                # if (i % config.training.print_freq == 0 or i == num_elem) and print_on_rank:
                if (i % 50 == 0 or i == num_elem) and print_on_rank:
                    argmax = torch.argmax(output, dim=1).to(torch.float32)
                    print(
                        f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, ",
                        f"min: {argmax.min().item()}, std: {argmax.std().item()}",
                    )
                    progress.display(i + 1, log=log)

                    # if argmax.std() == 0:
                    #     log.error(f"ISSUE WITH NETWORK printing debugging info")
                    #     for n, p in model.named_parameters():
                    #         print(f"{n}: {p.mean():.4f}, {p.min():.4f},
                    #         {p.max():.4f}, {p.std():.4f}")
                    #     raise ValueError

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE, pg=pg)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE, pg=pg)
    losses = AverageMeter("Loss", ":.4f", Summary.AVERAGE, pg=pg)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE, pg=pg)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE, pg=pg)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()
    # if epoch == 5:
    #     for n, p in model.named_parameters():
    #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
    #     raise ValueError
    vt1 = time.perf_counter()
    run_validate(val_loader)

    if len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * config["world_size"], len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=config.data["local_batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))
    val_time_total = time.perf_counter() - vt1

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    progress.display_summary(log=log, printing_rank=print_on_rank)

    if config["rank"] == 0 and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        wandb_logger.log(
            {
                "val/loss": ls,
                "val/top1": t1,
                "val/top5": t5,
                "val/total_time": val_time_total,
            },
            # step=(epoch + 1) * len_train,
        )
        # mlflow.log_metrics(
        #     metrics={
        #         "val loss": ls,
        #         "val top1": t1,
        #         "val top5": t5,
        #     },
        #     step=epoch,
        # )
    if config.rank == 0:
        log.info(f"Data loading time avg: {data_time.avg}")

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE, pg=None):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
        self.pg = pg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False, group=self.pg)
        self.sum, self.count = total.tolist()
        # self.avg = self.sum / self.count
        self.avg = total[0] / total[1]  # self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        # fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def display(self, batch, log):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # if self.rank == 0:
        #     # log.info("\t".join(entries))
        log.info(" ".join(entries))

    def display_summary(self, log, printing_rank):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if printing_rank:
            # print(" ".join(entries))
            # console.print(" ".join(entries))
            log.info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,), mixup=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        if not mixup:
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        else:
            maxk = max(topk)
            batch_size = target.size(0)
            if target.ndim == 2:
                target = target.max(dim=1)[1]

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k * (100.0 / batch_size))
            return res
