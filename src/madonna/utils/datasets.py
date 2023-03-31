from __future__ import annotations

from pathlib import Path

import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from omegaconf import DictConfig
from timm.data.transforms_factory import create_transform

__all__ = ["get_dataset"]
# from timm.data import


from PIL import ImageFile

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
        group_size: int=None, 
        group_rank: int=None, 
        num_groups: int=None
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
        config, group_size=group_size, group_rank=group_rank, num_groups=num_groups
    )

    return dset_dict


def get_imagenet_datasets(config, group_size=None, group_rank=None, num_groups=None):
    train_dataset, train_loader, train_sampler = imagenet_train_dataset_plus_loader(
        config=config, group_size=group_size, group_rank=group_rank, num_groups=num_groups
    )
    val_dataset, val_loader = imagenet_get_val_dataset_n_loader(
        config=config, group_size=group_size, group_rank=group_rank, num_groups=num_groups
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
    train_dataset, train_loader, train_sampler = cifar10_train_dataset_plus_loader(config, group_size=group_size, group_rank=group_rank, num_groups=num_groups)
    val_dataset, val_loader = cifar10_val_dataset_n_loader(config, group_size=group_size, group_rank=group_rank, num_groups=num_groups)
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
    train_dataset, train_loader, train_sampler = cifar100_train_dataset_plus_loader(config, group_size=group_size, group_rank=group_rank, num_groups=num_groups)
    val_dataset, val_loader = cifar100_val_dataset_n_loader(config, group_size=group_size, group_rank=group_rank, num_groups=num_groups)
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
    train_dataset, train_loader, train_sampler = mnist_train_data(config, group_size=group_size, group_rank=group_rank, num_groups=num_groups)
    val_dataset, val_loader = mnist_val_data(config, group_size=group_size, group_rank=group_rank, num_groups=num_groups)
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


def imagenet_train_dataset_plus_loader(
        config, group_size=None, group_rank=None, num_groups=None
    ):
    dsconfig = config["data"]
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
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    else:
        transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(train_crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    imagenet_normalize,
                ],
            ),
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
    )

    return train_dataset, train_loader, train_sampler


def imagenet_get_val_dataset_n_loader(
        config, group_size=None, group_rank=None, num_groups=None
    ):
    dsconfig = config["data"]
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    val_dir = Path(base_dir) / "val"
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(232),
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
            seed=dist.get_rank() // group_size
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
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    train_dir = Path(base_dir) / "train"

    if dsconfig["timm_transforms"]:
        transform = create_transform(
            32,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )
    else:
        transform = (
            transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32),
                    transforms.ToTensor(),
                    cifar10_normalize,
                ],
            ),
        )

    train_dataset = datasets.CIFAR10(
        root=str(train_dir),
        train=True,
        transform=transform,
        download=True,
    )

    # Data loader
    if dist.is_initialized():
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
    )
    return train_dataset, train_loader, train_sampler


def cifar10_val_dataset_n_loader(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]
    val_dir = Path(base_dir) / "val"

    test_dataset = datasets.CIFAR10(
        root=str(val_dir),
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), cifar10_normalize]),
    )

    if dist.is_initialized() and dsconfig["distributed_sample_val"]:
        sampler = datadist.DistributedSampler(test_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        sampler = datadist.DistributedSampler(
            test_dataset, 
            rank=group_rank, 
            num_replicas=group_size, 
            seed=dist.get_rank() // group_size
        )
    else:
        sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=dsconfig["num_workers"],
        sampler=sampler,
        persistent_workers=dsconfig["persistent_workers"],
    )
    return test_dataset, test_loader


def cifar100_train_dataset_plus_loader(config, group_size=None, group_rank=None, num_groups=None):
    # CIFAR-10 dataset
    dsconfig = config["data"]
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]

    train_dir = Path(base_dir) / "train"

    if dsconfig["timm_transforms"]:
        transform = create_transform(
            32,
            is_training=True,
            auto_augment="rand-m9-mstd0.5",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                cifar10_normalize,
            ],
        )

    train_dataset = datasets.CIFAR100(
        root=str(train_dir),
        train=True,
        transform=transform,  # timm_transforms,
        # download=True,
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
        pin_memory=dsconfig["num_workers"],
        num_workers=workers,
        persistent_workers=dsconfig["persistent_workers"],
    )
    return train_dataset, train_loader, train_sampler


def cifar100_val_dataset_n_loader(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]
    val_dir = Path(base_dir) / "val"

    test_dataset = datasets.CIFAR100(
        root=str(val_dir),
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), cifar10_normalize]),
        # download=True,
    )

    if dist.is_initialized() and dsconfig["distributed_sample_val"]:
        test_sampler = datadist.DistributedSampler(test_dataset)
    elif dist.is_initialized() and group_size is not None and group_size > 1:
        test_sampler = datadist.DistributedSampler(
            test_dataset, 
            rank=group_rank, 
            num_replicas=group_size, 
            seed=dist.get_rank() // group_size
        )
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=dsconfig["num_workers"],
        sampler=test_sampler,
        persistent_workers=dsconfig["persistent_workers"],
    )
    return test_dataset, test_loader


def mnist_train_data(config, group_size=None, group_rank=None, num_groups=None):
    dsconfig = config["data"]
    channels = config.model.mnist_channels if "mnist_channels" in config.model else 3
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]
    if dsconfig["resize"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=32),
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
    base_dir = dsconfig["data_dir"]
    batch_size = dsconfig["local_batch_size"]
    workers = dsconfig["num_workers"]
    if dsconfig["resize"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=32),
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
            seed=dist.get_rank() // group_size
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
