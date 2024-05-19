from __future__ import annotations

import os
import random
import shutil
import time
from enum import Enum
from pathlib import Path

import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
import madonna

# console = Console(width=140)
best_acc1 = 0


def main(config, log):  # noqa: C901
    import mlflow.pytorch

    if "seed" in config:
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])

    if dist.is_initialized():
        config["gpu"] = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
        log.debug(f"Using GPU: {config['gpu']}")
    else:
        config["gpu"] = 0
    torch.cuda.set_device(config["gpu"])
    device = torch.device(f"cuda:{config['gpu']}")

    model = madonna.utils.get_model(config)
    model.cuda(config["gpu"])
    if dist.is_initialized() and config.training.init_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = madonna.utils.get_criterion(config)
    optimizer = madonna.utils.get_optimizer(config, model)
    scheduler, warmup_scheduler = madonna.utils.get_lr_schedules(config, optimizer, 10)

    # optionally resume from a checkpoint
    # TODO: add checkpointing
    if "resume" in config:
        if os.path.isfile(config["resume"]):
            print(f"=> loading checkpoint: {config['resume']}")
            if config["gpu"] is None:
                checkpoint = torch.load(config["resume"])
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{config['gpu']}"
                checkpoint = torch.load(config["resume"], map_location=loc)
            config["start_epoch"] = checkpoint["epoch"]
            # best_acc1 = checkpoint["best_acc1"]
            # if config["gpu"] is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(config["gpu"])
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config["resume"],
                    checkpoint["epoch"],
                ),
            )
        else:
            print(f"=> no checkpoint found at: {config['resume']}")

    dset_dict = madonna.utils.datasets.get_dataset(config)
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    # if config['evaluate']:
    #     validate(val_loader, dlrt_trainer, config)
    #     return

    rlrp = False
    if isinstance(scheduler, ReduceLROnPlateau):
        rlrp = True
        epoch_step = True
    elif config["lr_schedule"]["name"] in [
        "CosineAnnealingWarmRestarts",
        "CosineAnnealingLR",
    ]:
        epoch_step = False
    else:  # StepLR / ExponentialLR / others
        epoch_step = True

    scaler = torch.cuda.amp.GradScaler(enabled=config.model.autocast)

    for epoch in range(config["start_epoch"], config["epochs"]):
        if config["rank"] == 0:
            # console.rule(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            log.info(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            mlflow.log_metrics(
                metrics={"lr": optimizer.param_groups[0]["lr"]},
                step=epoch,
            )
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        train_loss = train(
            train_loader,
            optimizer,
            model,
            criterion,
            epoch,
            device,
            config,
            warmup_scheduler=warmup_scheduler,
            log=log,
            lr_scheduler=scheduler,
            scaler=scaler,
        )

        # save_selected_weights(model, epoch)
        if config.rank == 0:
            log.info(f"Average Training loss across process space: {train_loss}")
        # evaluate on validation set
        _, val_loss = validate(val_loader, model, criterion, config, epoch)
        if config.rank == 0:
            log.info(
                f"Average val loss across process space: {val_loss} " f"-> diff: {train_loss - val_loss}",
            )
        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if rlrp:  # ReduceLROnPlateau requires the training loss
                    scheduler.step(train_loss)
                elif epoch_step:  # StepLR / ExponentialLR / others
                    scheduler.step()
        elif scheduler is not None:
            if rlrp:  # ReduceLROnPlateau requires the training loss
                scheduler.step(train_loss)
            elif epoch_step:  # StepLR / ExponentialLR / others
                scheduler.step()


def train(
    train_loader,
    optimizer,
    model,
    criterion,
    epoch,
    device,
    config,
    lr_scheduler,
    warmup_scheduler,
    log,
    scaler,
):
    import mlflow.pytorch

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
    if lr_scheduler is not None and config.lr_schedule._target_.split(".")[0] in [
        "CosineAnnealingWarmRestarts",
        "CosineAnnealingLR",
    ]:
        batch_lr_step = True
    else:
        batch_lr_step = False
    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.model.autocast):
            output = model(images)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

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
                if batch_lr_step:
                    lr_scheduler.step()
        elif lr_scheduler is not None and batch_lr_step:
            lr_scheduler.step()

        if (i % config["print_freq"] == 0 or i == len(train_loader) - 1) and config["rank"] == 0:
            # console.rule(f"train step {i}")
            argmax = torch.argmax(output, dim=1).to(torch.float32)
            # console.print(
            log.info(
                f"Argmax outputs s "
                f"mean: {argmax.mean().item():.5f}, max: {argmax.max().item():.5f}, "
                f"min: {argmax.min().item():.5f}, std: {argmax.std().item():.5f}",
            )
            progress.display(i + 1, log=log)

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if config["rank"] == 0:
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
    return losses.avg


@torch.no_grad()
def save_selected_weights(network, epoch):
    save_list = [
        "module.conv1.weight",
        "module.fc.weight",
        "module.layer1.1.conv2.weight",
        "module.layer3.1.conv2.weight",
        "module.layer4.0.downsample.0.weight",
    ]
    save_location = Path(
        "/hkfs/work/workspace/scratch/CHANGE/ME-dlrt/saved_models/4gpu-svd-tests/normal/resnet18",
    )
    rank = dist.get_rank()

    if rank != 0:
        dist.barrier()
        return
    # save location: resnet18/name/epoch/[u, s, vh, weights
    for n, p in network.named_parameters():
        if n in save_list:
            print(n)
            n_save_loc = save_location / n / str(epoch)
            n_save_loc.mkdir(exist_ok=True, parents=True)
            # todo: full matrices?? -> can always slice later
            if p.data.ndim > 2:
                tosave = p.view(p.shape[0], -1)
            else:
                tosave = p.data
            u, s, vh = torch.linalg.svd(tosave, full_matrices=False)
            torch.save(u, n_save_loc / "u-reduced.pt")
            torch.save(s, n_save_loc / "s-reduced.pt")
            torch.save(vh, n_save_loc / "vh-reduced.pt")
            u, s, vh = torch.linalg.svd(tosave, full_matrices=True)
            torch.save(u, n_save_loc / "u.pt")
            torch.save(s, n_save_loc / "s.pt")
            torch.save(vh, n_save_loc / "vh.pt")
            torch.save(tosave.data, n_save_loc / "p.pt")
    print("finished saving")
    dist.barrier()


def validate(val_loader, model, criterion, config, epoch, log):
    # console.rule("validation")

    def run_validate(loader, base_progress=0):
        rank = 0 if not dist.is_initialized() else dist.get_rank()
        with torch.no_grad():
            end = time.time()
            num_elem = len(loader) - 1
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(config["gpu"], non_blocking=True)
                target = target.cuda(config["gpu"], non_blocking=True)

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

                if (i % config["print_freq"] == 0 or i == num_elem) and rank == 0:
                    argmax = torch.argmax(output, dim=1).to(torch.float32)
                    print(
                        f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, ",
                        f"min: {argmax.min().item()}, std: {argmax.std().item()}",
                    )
                    progress.display(i + 1, log=log)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4f", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

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

    progress.display_summary(log=log)

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if config["rank"] == 0:
        import mlflow.pytorch

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

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

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
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
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

    def display_summary(self, log):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if self.rank == 0:
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
