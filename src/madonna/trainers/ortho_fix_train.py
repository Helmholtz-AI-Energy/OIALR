from __future__ import annotations

import logging
import os
import random
import shutil
import time
from enum import Enum
from pathlib import Path

import hydra
import mlflow.pytorch

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
import omegaconf
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
from torch.utils.data import Subset

import madonna

best_acc1 = 0
log = logging.getLogger(__name__)


def main(config):  # noqa: C901
    if config.seed is not None:
        cudnn.benchmark = True
        cudnn.deterministic = True
        # torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=5)
        # if dist.is_initialized():
        #     scale = dist.get_rank() % 4
        # else:
        #     scale = 0
        random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))

    if config.cpu_training:
        gpu = None
        log.debug("NOT using GPUs!")
        device = torch.device("cpu")
    else:
        if dist.is_initialized():
            gpu = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
            log.debug(f"Using GPU: {gpu}")
        else:
            gpu = 0
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    # print('before get model')
    model = madonna.utils.get_model(config)
    if not config.cpu_training:
        model.cuda(gpu)
    # print('before model wrapping')

    if not config.baseline:
        # model = madonna.models.QRFixingModel(model, **config.training.qr_fixing)
        model_hold = hydra.utils.instantiate(config.training.fixing_method)
        model = model_hold(model).to(device)
        # model = madonna.models.QROrthoFixingModel(model, **config.training.fixing_method)
    elif dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model)  # , device_ids=[config.rank])
        if dist.get_rank() == 0:
            print(model)
    else:
        print(model)

    # for n, p in model.named_parameters():
    #     print(f"{n}, {p.shape}")
    # raise ValueError

    criterion = madonna.utils.get_criterion(config)
    optimizer = madonna.utils.get_optimizer(config, model)  # -> madonna.optimizers.myopt.MyOpt

    scheduler, warmup_scheduler = madonna.utils.get_lr_schedules(config, optimizer, 10)

    # # optionally resume from a checkpoint
    # # TODO: add checkpointing
    # # Reminder: when resuming from a single checkpoint, make sure to call init_model with
    # # keep_rank0=True
    # if "resume" in config:
    #     if os.path.isfile(config["resume"]):
    #         print(f"=> loading checkpoint: {config['resume']}")
    #         if config["gpu"] is None:
    #             checkpoint = torch.load(config["resume"])
    #         elif torch.cuda.is_available():
    #             # Map model to be loaded to specified single gpu.
    #             loc = f"cuda:{config['gpu']}"
    #             checkpoint = torch.load(config["resume"], map_location=loc)
    #         config["start_epoch"] = checkpoint["epoch"]
    #         # best_acc1 = checkpoint["best_acc1"]
    #         # if config["gpu"] is not None:
    #         #     # best_acc1 may be from a checkpoint from a different GPU
    #         #     best_acc1 = best_acc1.to(config["gpu"])
    #         model.load_state_dict(checkpoint["state_dict"])
    #         # optimizer.load_state_dict(checkpoint["optimizer"])
    #         scheduler.load_state_dict(checkpoint["scheduler"])
    #         print(
    #             "=> loaded checkpoint '{}' (epoch {})".format(
    #                 config["resume"],
    #                 checkpoint["epoch"],
    #             ),
    #         )
    #     else:
    #         print(f"=> no checkpoint found at: {config['resume']}")

    # if optimizer.num_groups > 1:
    #     dset_dict = madonna.utils.datasets.get_dataset(
    #         config,
    #         group_size=optimizer.local_size,
    #         group_rank=optimizer.local_rank,
    #         num_groups=optimizer.num_groups,
    #     )
    # else:
    dset_dict = madonna.utils.datasets.get_dataset(config)
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    # # if config['evaluate']:
    # #     validate(val_loader, dlrt_trainer, config)
    # #     return
    #
    if warmup_scheduler is not None and config.training.lr_schedule._target_.split(".")[0] in [
        "CosineAnnealingWarmRestarts",
        "CosineAnnealingLR",
    ]:
        batch_warmup_step = True
    else:
        batch_warmup_step = False

    scaler = torch.cuda.amp.GradScaler(enabled=config.model.autocast)
    rank = dist.get_rank() if dist.is_initialized() else 0
    # try:
    #     warmup_steps = config.training.lr_warmup.warmup_period
    # except omegaconf.errors.ConfigAttributeError:
    #     log.info("No number of warmup steps specified")

    is_adam = isinstance(optimizer, torch.optim.Adam)
    is_sgd = isinstance(optimizer, torch.optim.SGD)
    first = True
    for epoch in range(config.training["start_epoch"], config.training["epochs"]):
        if config["rank"] == 0:
            # console.rule(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            log.info(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            mlflow.log_metrics(
                metrics={"lr": optimizer.param_groups[0]["lr"]},
                step=epoch,
            )
        if dist.is_initialized() and config.data.distributed_sample and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        with torch.no_grad():
            if epoch > config.training.svd_epoch_delay + 1:
                for n, p in model.named_parameters():
                    if n.endswith(".s"):
                        # # TODO: remove later if not working
                        # sdiag = torch.diag(self.s).clone()
                        # sdiag = torch.diag(p)
                        # sdiag_diff1 = torch.diff(sdiag, n=1)
                        # sdiag_diff2 = torch.diff(sdiag, n=2)
                        # sdiag_diff3 = torch.diff(sdiag, n=3)

                        mask = torch.abs(p) <= 1e-7
                        # print(f"{n} -> {torch.count_nonzero(mask)}")
                        p[mask] *= 0
                        p[mask] += 1e-4 * torch.rand_like(p[mask])  # * sdiag.max()

        train_loss, last_loss, optimizer = train(
            train_loader=train_loader,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            epoch=epoch,
            device=device,
            config=config,
            scaler=scaler,
            lr_scheduler=scheduler,
            warmup_scheduler=warmup_scheduler,
        )

        # if epoch * len(train_loader) >= warmup_steps or epoch == config.training.epochs - 1:  # epoch % 2 == 1
        if epoch > config.training.svd_epoch_delay or epoch == config.training.epochs - 1:
            # try:
            # dist.barrier()
            optimizer.zero_grad(set_to_none=True)
            stabtime = time.perf_counter()
            reset_optimizer = model.test_basis_stability_all_layers()
            if rank == 0:
                log.info(f"Stability time: {time.perf_counter() - stabtime}")
            # except AttributeError:
            #     reset_optimizer = False
            if reset_optimizer:
                if is_adam:
                    madonna.optimizers.change_adam_shapes(
                        optimizer=optimizer,
                        model=model,
                        reset_buffers_zero=first,
                    )
                elif is_sgd:
                    madonna.optimizers.change_sgd_shapes(
                        optimizer=optimizer,
                        model=model,
                        reset_buffers_zero=first,
                    )
                first = False

                # resettime = time.perf_counter()
                # # instead of resetting optimizer, slice off bits of the saved states
                # # for group in optimizer.param_groups:
                # for c, (n, p) in enumerate(model.named_parameters()):
                #     # if dist.get_rank() == 0:
                #     #     print(n, optimizer.param_groups[0]["params"][c].shape, p.shape)
                #     if is_adam:
                #         state = optimizer.state[p]
                #         if len(list(state.keys())) > 0:
                #             for k in ["exp_avg", "exp_avg_sq"]:
                #                 if state[k].shape != p.shape:
                #                     sl = []
                #                     for d in range(p.ndim):
                #                         sl.append(slice(0, p.shape[d]))
                #                     # print(type(state[k]))
                #                     state[k] = state[k][tuple(sl)]
                #             if optimizer.param_groups[0]["amsgrad"]:
                #                 if state["max_exp_avg_sq"].shape != p.shape:
                #                     sl = []
                #                     for d in range(p.ndim):
                #                         sl.append(slice(0, p.shape[d]))
                #                     state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
                #     if optimizer.param_groups[0]["params"][c].shape != p.shape:
                #         sl = []
                #         for d in range(p.ndim):
                #             sl.append(slice(0, p.shape[d]))
                #         optimizer.param_groups[0]["params"][c] = optimizer.param_groups[0]["params"][c][tuple(sl)]
                # if rank == 0:
                #     log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")
                # optimizer = madonna.utils.get_optimizer(config, model.ddp_model, lr=optimizer.param_groups[0]['lr'])
                # # if resetting optimizer, need to also reset the lr scheduler and warmup
                # scheduler.optimizer = optimizer
                # warmup_scheduler.optimizer = optimizer

        # if model.skip_stability:
        # for n, p in model.named_parameters():
        #     print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        # model.sync_models()
        # if epoch % 10 == 9 or epoch == config.training.epochs - 1:
        #     if dist.get_rank() == 0:
        #         for n, p in model.named_parameters():
        #             print(f"{n} {p.requires_grad} sparcity: {torch.count_nonzero(torch.abs(p) < 1e-5) / p.numel()}")

        if config.rank == 0:
            log.info(f"Average Training loss across process space: {train_loss}")
        # evaluate on validation set
        _, val_loss = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            config=config,
            epoch=epoch,
            device=device,
            print_on_rank=config.rank == 0,
            pg=None,
        )

        if config.rank == 0:
            log.info(
                f"Average val loss across process space: {val_loss} " f"-> diff: {train_loss - val_loss}",
            )

        # log the percentage of params in use
        if config.rank == 0 and not config.baseline:
            if hasattr(model, "get_perc_params_all_layers"):
                perc, trainable, normal = model.get_perc_params_all_layers(module=model.ddp_model)
                if config.enable_tracking:
                    mlflow.log_metric("perc_parmas", perc, step=epoch)
                log.info(f"% params: {perc:.3f}% trainable: {trainable} full: {normal}")
                model.track_interior_slices_mlflow(config, epoch)

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if not batch_warmup_step:
                    scheduler.step()
        elif scheduler is not None and not batch_warmup_step:
            scheduler.step()


def train(
    train_loader,
    optimizer,
    model,
    criterion,
    epoch,
    device,
    config,
    lr_scheduler=None,
    warmup_scheduler=None,
    scaler=None,
    log=log,
):
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
    if warmup_scheduler is not None and config.training.lr_schedule._target_.split(".")[0] in [
        "CosineAnnealingWarmRestarts",
        "CosineAnnealingLR",
    ]:
        batch_warmup_step = True
    else:
        batch_warmup_step = False
    # switch to train mode
    model.train()
    # madonna.utils.change_batchnorm_tracking(model, tracking=False)
    end = time.time()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        if hasattr(config.data, "dali") and config.data.dali:
            images = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        else:
            images = data[0]
            target = data[1]
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # try:
        #     reset_optimizer = model.check_stability()
        # except AttributeError:
        #     reset_optimizer = False
        # if reset_optimizer:
        #     if dist.get_rank() == 0:
        #         log.info("Reset Optimizer State")
        #     optimizer = madonna.utils.get_optimizer(config, model.ddp_model, lr=optimizer.param_groups[0]['lr'])
        #     # if resetting optimizer, need to also reset the lr scheduler and warmup
        #     lr_scheduler.optimizer = optimizer
        #     warmup_scheduler.optimizer = optimizer

        # if not config.baseline and dist.is_initialized():
        #     with model.target_model.no_sync():
        #         with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.model.autocast):
        #             output = model(images)
        #             loss = criterion(output, target)

        #         if torch.isnan(loss):
        #             for n, p in model.named_parameters():
        #                 print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        #             raise ValueError("NaN loss")

        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)
        #         scaler.update()
        # else:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.model.autocast):
            output = model(images)
            loss = criterion(output, target)

        if torch.isnan(loss):
            for n, p in model.named_parameters():
                print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
            raise ValueError("NaN loss")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # for n, p in model.named_parameters():
        #     print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}, {p.requires_grad}")
        # optimizer.step()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if batch_warmup_step:
                    lr_scheduler.step()

        # if argmax.std() == 0:
        #     log.error(f"ISSUE WITH NETWORK printing debugging info")
        #     for n, p in model.named_parameters():
        #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        #     raise ValueError
        argmax = torch.argmax(output, dim=1).to(torch.float32)
        # if argmax.std() == 0 and epoch > 5:
        #     if dist.get_rank() == 0:
        #         for n, p in model.named_parameters():
        #             print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        #     raise ValueError("Std == 0")

        if (i % config.training.print_freq == 0 or i == len(train_loader) - 1) and config["rank"] == 0:
            argmax = torch.argmax(output, dim=1).to(torch.float32)
            log.info(
                f"Argmax outputs s "
                f"mean: {argmax.mean().item():.5f}, max: {argmax.max().item():.5f}, "
                f"min: {argmax.min().item():.5f}, std: {argmax.std().item():.5f}",
            )
            progress.display(i + 1, log=log)

            # if argmax.std() == 0:
            #     log.error(f"ISSUE WITH NETWORK printing debugging info")
            #     for n, p in model.named_parameters():
            #         print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
            #     raise ValueError
            # dist.barrier()
        # if i == 60:
        #     raise RuntimeError
    if config.rank == 0:
        log.info(f"Data Loading Time avg: {data_time.avg}")

    # if not config.baseline:
    #     model.sync_models()

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if config["rank"] == 0 and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        mlflow.log_metrics(
            metrics={
                "train loss": ls,
                "train top1": t1,
                "train top5": t5,
            },
            step=epoch,
        )
    return losses.avg, loss, optimizer


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
def validate(val_loader, model, criterion, config, epoch, device, print_on_rank, pg):
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

                # compute output
                output = model(images)
                loss = criterion(output, target)
                # argmax = torch.argmax(output.output, dim=1).to(torch.float32)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
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

    run_validate(val_loader)

    if len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * config["world_size"], len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=config["local_batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    progress.display_summary(log=log, printing_rank=print_on_rank)

    if config["rank"] == 0 and config.enable_tracking:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        mlflow.log_metrics(
            metrics={
                "val loss": ls,
                "val top1": t1,
                "val top5": t5,
            },
            step=epoch,
        )
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
