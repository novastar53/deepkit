import numpy as np
import jax.numpy as jnp
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils import data

__all__ = ["load_CIFAR10"]

# CIFAR-10


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.array(pic.permute(1, 2, 0), dtype=jnp.float32)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def load_CIFAR10(batch_size=128, augment=True):

    train_transforms = [transforms.ToTensor()]

    if augment:
        train_transforms += [
            transforms.RandomCrop((32, 32), padding=4, fill=0, padding_mode="constant"),
            transforms.RandomHorizontalFlip(),
        ]

    train_transforms += [
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
        FlattenAndCast(),
    ]

    transforms_train = transforms.Compose(train_transforms)

    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
            FlattenAndCast(),
        ]
    )

    train_dataset = CIFAR10(
        root="./CIFAR", train=True, download=True, transform=transforms_train
    )

    test_dataset = CIFAR10(
        root="./CIFAR", train=False, download=True, transform=transforms_test
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = NumpyLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, test_loader
