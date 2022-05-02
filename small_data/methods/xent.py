import numpy as np
import torch
from torch import nn
from typing import Callable
import torchvision.transforms as tf
import torch.utils.data as datautil

from PIL import Image
from typing import Tuple, Callable, Optional, Union
from ..utils import is_notebook
from abc import abstractmethod

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

from .common import BasicAugmentation
from .common import LearningMethod


class My_DataAugmentation(LearningMethod):
    def get_data_transforms(self, dataset) -> Tuple[Callable, Callable]:

        transforms = []
        test_transforms = []

        # Check whether data type is PIL image or NumPy array
        # (arrays must be converted to tensors before anything else)
        pil_data = isinstance(dataset[0][0], Image.Image)
        if not pil_data:
            transforms.append(tf.ToTensor())
            test_transforms.append(tf.ToTensor())

        # Resize/Crop/Shift
        if self.hparams["target_size"] is not None:
            if self.hparams["rand_shift"] > 0:
                transforms.append(
                    tf.Resize(
                        self.hparams["target_size"] + 2 * self.hparams["rand_shift"]
                    )
                )
                transforms.append(tf.RandomCrop(self.hparams["target_size"]))
            else:
                transforms.append(tf.Resize(self.hparams["target_size"]))
            test_transforms.append(
                tf.Resize(self.hparams["target_size"] + 2 * self.hparams["rand_shift"])
            )
            test_transforms.append(tf.CenterCrop(self.hparams["target_size"]))
        elif self.hparams["rand_shift"] > 0:
            img_size = np.asarray(dataset[0][0]).shape[:2]
            transforms.append(
                tf.RandomCrop(
                    img_size, padding=self.hparams["rand_shift"], padding_mode="reflect"
                )
            )

        # Horizontal/Vertical Flip
        if self.hparams["hflip"]:
            transforms.append(tf.RandomHorizontalFlip())
        if self.hparams["vflip"]:
            transforms.append(tf.RandomVerticalFlip())

        # transforms.append(tf.RandomPerspective(distortion_scale=0.6, p=1.0))
        #transforms.append(tf.RandomRotation(degrees=(-20, 20)))
        # transforms.append(tf.RandomAffine(degrees=(0,0), translate=(0.1, 0.3), shear=2))
        
        transforms.append(tf.AutoAugment(tf.AutoAugmentPolicy.CIFAR10))

        # Convert PIL image to tensor
        if pil_data:
            transforms.append(tf.ToTensor())
            test_transforms.append(tf.ToTensor())

        # Channel-wise normalization
        if self.hparams["normalize"]:
            channel_mean, channel_std = None, None
            if not self.hparams["recompute_statistics"]:
                try:
                    channel_mean, channel_std = dataset.get_normalization_statistics()
                except AttributeError:
                    pass
            if channel_mean is None:
                channel_mean, channel_std = self.compute_normalization_statistics(
                    dataset
                )
            norm_trans = tf.Normalize(channel_mean, channel_std)
            transforms.append(norm_trans)
            test_transforms.append(norm_trans)

        return tf.Compose(transforms), tf.Compose(test_transforms)

    def compute_normalization_statistics(
        self, dataset, show_progress=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes channel-wise mean and standard deviation for a given dataset.

        Parameters
        ----------
        dataset : small_data.datasets.ImageClassificationDataset
            The dataset.
        show_progress : bool
            Whether to show a tqdm progress bar.

        Returns
        -------
        mean : np.ndarray
        std : np.ndarray
        """

        # Check whether data type is PIL image or NumPy array
        pil_data = isinstance(dataset[0][0], Image.Image)

        # Create data loader with resize and center crop transform
        transforms = []
        if not pil_data:
            transforms.append(tf.ToTensor())
        if self.hparams["target_size"]:
            transforms.append(tf.Resize(self.hparams["target_size"]))
            transforms.append(tf.CenterCrop(self.hparams["target_size"]))
        if pil_data:
            transforms.append(tf.ToTensor())
        prev_transform = dataset.transform
        dataset.transform = tf.Compose(transforms)
        data_loader = datautil.DataLoader(dataset, batch_size=1000, shuffle=False)

        # Compute mean
        num_samples = 0
        channel_mean = 0
        for batch, _ in tqdm(
            data_loader, desc="Computing mean", disable=not show_progress
        ):
            channel_mean = channel_mean + batch.sum(axis=(0, 2, 3))
            num_samples += batch.shape[0] * batch.shape[2] * batch.shape[3]
        channel_mean /= num_samples

        # Compute standard deviation
        channel_std = 0
        for batch, _ in tqdm(
            data_loader, desc="Computing std", disable=not show_progress
        ):
            batch -= channel_mean[None, :, None, None]
            channel_std = channel_std + (batch * batch).sum(axis=(0, 2, 3))
        channel_std = torch.sqrt(channel_std / num_samples)

        # Restore saved transform
        dataset.transform = prev_transform

        return channel_mean.numpy().copy(), channel_std.numpy().copy()

    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(BasicAugmentation, BasicAugmentation).default_hparams(),
            "normalize": True,
            "recompute_statistics": False,
            "target_size": None,
            "min_scale": 1.0,
            "max_scale": 1.0,
            "rand_shift": 0,
            "hflip": True,
            "vflip": False,
        }


from typing import Iterable

import torch
from torch.optim._multi_tensor import SGD

__all__ = ["SAMSGD"]


class CrossEntropyClassifier(My_DataAugmentation):
    """Standard cross-entropy classification as baseline.

    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def get_loss_function(self) -> Callable:

        # return nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
        # return nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='none')
        return nn.CrossEntropyLoss(reduction="mean")

    def get_optimizer(
        self, model: nn.Module, max_epochs: int, max_iter: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Instantiates an optimizer and learning rate schedule.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        max_epochs : int
            The total number of epochs.
        max_iter : int
            The total number of iterations (epochs * batches_per_epoch).

        Returns
        -------
        optimizer : torch.optim.Optimizer
        lr_schedule : torch.optim.lr_scheduler._LRScheduler
        """

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.hparams["lr"],
            momentum=0.9,
            weight_decay=self.hparams["weight_decay"],
        )

        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=self.hparams["lr"],
        #     betas=(0.9, 0.999),
        #     eps=1e-08,
        #     weight_decay=self.hparams["weight_decay"],
        #     amsgrad=False,
        # )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iter
        )
        return optimizer, scheduler
