from __future__ import annotations

import logging
import socket
from pathlib import Path

import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import ImageFile
from timm.data.transforms_factory import create_transform

log = logging.getLogger(__name__)

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

    has_dali = True
except ImportError:
    has_dali = False

uc2 = socket.gethostname().startswith("uc2")
horeka = socket.gethostname().startswith("hkn")


__all__ = ["get_dataset"]
# from timm.data import

ImageFile.LOAD_TRUNCATED_IMAGES = True


imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

cifar10_normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)
mnist_normalize = transforms.Normalize(
    mean=(0.1307,),
    std=(0.3081,),
)


def get_dataset(
    config: DictConfig,
    group_size: int = None,
    group_rank: int = None,
    num_groups: int = None,
) -> dict:
    # get datasets
    options = {
        "imagenet": get_imagenet_datasets,
        "cifar10": get_cifar10_datasets,
        "cifar100": get_cifar100_datasets,
        "mnist": get_mnist_datasets,
    }
    if config.data.dataset not in options:
        raise ValueError("dataset not in options! add to utils.datasets")

    dset_dict = options[config.data.dataset](
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )

    return dset_dict


def get_imagenet_datasets(config, group_size=None, group_rank=None, num_groups=None):
    train_dataset, train_loader, train_sampler = imagenet_train_dataset_plus_loader(
        config=config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    val_dataset, val_loader = imagenet_get_val_dataset_n_loader(
        config=config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    return {
        "train": {
            "dataset": train_dataset,
            "loader": train_loader,
            "sampler": train_sampler,
        },
        "val": {
            "dataset": val_dataset,
            "loader": val_loader,
        },
    }


def get_cifar10_datasets(config, group_size=None, group_rank=None, num_groups=None):
    train_dataset, train_loader, train_sampler = cifar10_train_dataset_plus_loader(
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    val_dataset, val_loader = cifar10_val_dataset_n_loader(
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    return {
        "train": {
            "dataset": train_dataset,
            "loader": train_loader,
            "sampler": train_sampler,
        },
        "val": {
            "dataset": val_dataset,
            "loader": val_loader,
        },
    }


def get_cifar100_datasets(config, group_size=None, group_rank=None, num_groups=None):
    train_dataset, train_loader, train_sampler = cifar100_train_dataset_plus_loader(
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    val_dataset, val_loader = cifar100_val_dataset_n_loader(
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    return {
        "train": {
            "dataset": train_dataset,
            "loader": train_loader,
            "sampler": train_sampler,
        },
        "val": {
            "dataset": val_dataset,
            "loader": val_loader,
        },
    }


def get_mnist_datasets(config, group_size=None, group_rank=None, num_groups=None):
    train_dataset, train_loader, train_sampler = mnist_train_data(
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    val_dataset, val_loader = mnist_val_data(
        config,
        group_size=group_size,
        group_rank=group_rank,
        num_groups=num_groups,
    )
    return {
        "train": {
            "dataset": train_dataset,
            "loader": train_loader,
            "sampler": train_sampler,
        },
        "val": {
            "dataset": val_dataset,
            "loader": val_loader,
        },
    }


if has_dali:

    @pipeline_def
    def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
        images, labels = fn.readers.file(
            file_root=data_dir,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=is_training,
            pad_last_batch=True,
            name="Reader",
        )
        dali_device = "cpu" if dali_cpu else "gpu"
        decoder_device = "cpu" if dali_cpu else "mixed"
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == "mixed" else 0
        preallocate_height_hint = 6430 if decoder_device == "mixed" else 0
        if is_training:
            images = fn.decoders.image_random_crop(
                images,
                device=decoder_device,
                output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                preallocate_width_hint=preallocate_width_hint,
                preallocate_height_hint=preallocate_height_hint,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100,
            )
            images = fn.resize(
                images,
                device=dali_device,
                resize_x=crop,
                resize_y=crop,
                interp_type=types.INTERP_TRIANGULAR,
            )
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(
                images,
                device=decoder_device,
                output_type=types.RGB,
            )
            images = fn.resize(
                images,
                device=dali_device,
                size=size,
                mode="not_smaller",
                interp_type=types.INTERP_TRIANGULAR,
            )
            mirror = False

        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=mirror,
        )
        labels = labels.gpu()
        return images, labels

    def _imagenet_dali_train(config):
        dsconfig = config["data"]
        if uc2:
            base_dir = config.data.data_dir_uc2
        elif horeka:
            base_dir = config.data.data_dir_horeka
        else:
            base_dir = dsconfig["data_dir"]
        batch_size = dsconfig["local_batch_size"]
        workers = dsconfig["num_workers"]

        train_dir = Path(base_dir) / "train"

        # TODO: figure out the timm transform stuff
        log.info("TODO: add timm transform options")

        train_crop_size = config.data.train_crop_size
        pipe = create_dali_pipeline(
            batch_size=batch_size,
            num_threads=workers,
            device_id=config.rank % 4,
            seed=12 + config.rank,
            data_dir=str(train_dir),
            crop=train_crop_size,
            size=train_crop_size,
            dali_cpu=False,
            shard_id=config.rank,
            num_shards=config.world_size,
            is_training=True,
        )
        pipe.build()
        train_loader = DALIClassificationIterator(
            pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        return None, train_loader, None

    def _imagenet_dali_val(config):
        dsconfig = config["data"]
        if uc2:
            base_dir = config.data.data_dir_uc2
        elif horeka:
            base_dir = config.data.data_dir_horeka
        else:
            base_dir = dsconfig["data_dir"]
        batch_size = dsconfig["local_batch_size"]
        workers = dsconfig["num_workers"]

        val_dir = Path(base_dir) / "val"

        # TODO: figure out the timm transform stuff
        log.info("TODO: add timm transform options")

        train_crop_size = config.data.train_crop_size
        pipe = create_dali_pipeline(
            batch_size=batch_size,
            num_threads=workers,
            device_id=config.rank % 4,
            seed=12 + config.rank,
            data_dir=str(val_dir),
            crop=train_crop_size,
            size=train_crop_size,
            dali_cpu=False,
            shard_id=config.rank,
            num_shards=config.world_size,
            is_training=False,
        )
        pipe.build()
        val_loader = DALIClassificationIterator(
            pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        return None, val_loader


def imagenet_train_dataset_plus_loader(
    config,
    group_size=None,
    group_rank=None,
    num_groups=None,
):
    dsconfig = config["data"]
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    if config.data.dali and has_dali:
        return _imagenet_dali_train(config)
    elif config.data.dali:
        raise ImportError("Attempt to use DALI but DALI not installed")

    train_dir = Path(base_dir) / "train"

    train_crop_size = config.data.train_crop_size

    if dsconfig["timm_transforms"]:
        transform = create_transform(
            train_crop_size,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalize,
            ],
        )

    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform,
    )

    if dist.is_initialized() and dsconfig["distributed_sample"]:
        train_sampler = datadist.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=dsconfig["num_workers"],
        sampler=train_sampler,
        persistent_workers=dsconfig["persistent_workers"],
        prefetch_factor=dsconfig.prefetch_factor,
    )

    return train_dataset, train_loader, train_sampler


def imagenet_get_val_dataset_n_loader(
    config,
    group_size=None,
    group_rank=None,
    num_groups=None,
):
    dsconfig = config["data"]
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    if config.data.dali and has_dali:
        return _imagenet_dali_val(config)
    elif config.data.dali:
        raise ImportError("Attempt to use DALI but DALI not installed")

    val_dir = Path(base_dir) / "val"
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_normalize,
            ],
        ),
    )

    if dist.is_initialized() and dsconfig["distributed_sample_val"]:
        val_sampler = datadist.DistributedSampler(val_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        val_sampler = datadist.DistributedSampler(
            val_dataset,
            rank=group_rank,
            num_replicas=group_size,
            seed=dist.get_rank() // group_size,
        )
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=dsconfig["num_workers"],
        sampler=val_sampler,
        persistent_workers=dsconfig["persistent_workers"],
    )

    return val_dataset, val_loader


def cifar10_train_dataset_plus_loader(config, group_size=None, group_rank=None, num_groups=None):
    # CIFAR-10 dataset
    dsconfig = config["data"]
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    train_dir = Path(base_dir) / "train"
    train_crop_size = config.data.train_crop_size

    if dsconfig["timm_transforms"]:
        transform = create_transform(
            train_crop_size,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )
    else:
        trans_list = [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(train_crop_size),
            transforms.ToTensor(),
            cifar10_normalize,
        ]
        trans_list.extend(
            [],
        )
        transform = transforms.Compose(trans_list)

    train_dataset = datasets.CIFAR10(
        root=str(train_dir),
        train=True,
        transform=transform,
        download=True,
    )

    # Data loader
    if dist.is_initialized() and config.data.distributed_sample:
        train_sampler = datadist.DistributedSampler(train_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        train_sampler = datadist.DistributedSampler(
            train_dataset,
            rank=group_rank,
            num_replicas=group_size,
            seed=dist.get_rank() // group_size,
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=False,  # dsconfig["num_workers"],
        sampler=train_sampler,
        persistent_workers=dsconfig["persistent_workers"],
    )
    return train_dataset, train_loader, train_sampler


def cifar10_val_dataset_n_loader(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    # workers = dsconfig["num_workers"]
    val_dir = Path(base_dir) / "val"

    trans = [
        transforms.ToTensor(),
        transforms.Resize(config.data.train_crop_size, antialias=True),
        cifar10_normalize,
    ]
    trans = transforms.Compose(trans)

    test_dataset = datasets.CIFAR10(
        root=str(val_dir),
        train=False,
        transform=trans,
        download=True,
    )

    if dist.is_initialized() and dsconfig["distributed_sample_val"]:
        sampler = datadist.DistributedSampler(test_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        sampler = datadist.DistributedSampler(
            test_dataset,
            rank=group_rank,
            num_replicas=group_size,
            seed=dist.get_rank() // group_size,
        )
    else:
        sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,  # dsconfig["num_workers"],
        sampler=sampler,
        # persistent_workers=dsconfig["persistent_workers"],
    )
    return test_dataset, test_loader


def cifar100_train_dataset_plus_loader(config, group_size=None, group_rank=None, num_groups=None):
    # CIFAR-10 dataset
    dsconfig = config["data"]
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    train_dir = Path(base_dir) / "train"

    if dsconfig["timm_transforms"]:
        transform = create_transform(
            config.data.train_crop_size,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )
    else:
        trans_list = [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.Resize(config.data.train_crop_size),
            transforms.ToTensor(),
            cifar10_normalize,
        ]
        transform = transforms.Compose(trans_list)

    train_dataset = datasets.CIFAR100(
        root=str(train_dir),
        train=True,
        transform=transform,  # timm_transforms,
        download=True,
    )

    # Data loader
    if dist.is_initialized() and dsconfig["distributed_sample"]:
        train_sampler = datadist.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,  # dsconfig["num_workers"],
        num_workers=workers,
        persistent_workers=dsconfig["persistent_workers"],
    )
    return train_dataset, train_loader, train_sampler


def cifar100_val_dataset_n_loader(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    # workers = dsconfig["num_workers"]
    val_dir = Path(base_dir) / "val"

    trans = [transforms.ToTensor(), transforms.Resize(config.data.train_crop_size)]
    trans.append(cifar10_normalize)
    trans = transforms.Compose(trans)

    test_dataset = datasets.CIFAR100(
        root=str(val_dir),
        train=False,
        transform=trans,
        download=True,
    )

    if dist.is_initialized() and dsconfig["distributed_sample_val"]:
        test_sampler = datadist.DistributedSampler(test_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        test_sampler = datadist.DistributedSampler(
            test_dataset,
            rank=group_rank,
            num_replicas=group_size,
            seed=dist.get_rank() // group_size,
        )
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,  # dsconfig["num_workers"],
        sampler=test_sampler,
        # persistent_workers=dsconfig["persistent_workers"],
    )
    return test_dataset, test_loader


def mnist_train_data(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    channels = config.model.mnist_channels if "mnist_channels" in config.model else 3
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]
    if dsconfig["resize"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(config.data.train_crop_size),
                transforms.Grayscale(channels),
                mnist_normalize,
            ],
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), mnist_normalize])
    train_dataset = datasets.MNIST(base_dir, train=True, download=True, transform=transform)

    if dist.is_initialized() and dsconfig["distributed_sample"]:
        train_sampler = datadist.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        persistent_workers=False,
    )
    return train_dataset, train_loader, train_sampler


def mnist_val_data(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    channels = config.model.mnist_channels if "mnist_channels" in config.model else 3
    if uc2:
        base_dir = config.data.data_dir_uc2
    elif horeka:
        base_dir = config.data.data_dir_horeka
    else:
        base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]
    if dsconfig["resize"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(config.data.train_crop_size),
                transforms.Grayscale(channels),
                mnist_normalize,
            ],
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), mnist_normalize])
    val_dataset = datasets.MNIST(base_dir, train=False, transform=transform)

    if dist.is_initialized() and dsconfig["distributed_sample_val"]:
        sampler = datadist.DistributedSampler(val_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        sampler = datadist.DistributedSampler(
            val_dataset,
            rank=group_rank,
            num_replicas=group_size,
            seed=dist.get_rank() // group_size,
        )
    else:
        sampler = None

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        sampler=sampler,
    )
    return val_dataset, val_loader


def vit_size(config, alt_size):
    if config.model.name.startswith("vit"):
        return 224
    return alt_size
