import math
import warnings
from bisect import bisect_right
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


def build_scheduler(cfg, optimizer, dataloader_len):
    scheduler_type = cfg["type"]
    cfg.pop("type")

    max_epoch = cfg["max_epoch"]

    if scheduler_type == "LinearWarmupCosineAnnealingLR":
        cfg["warmup_epoch"] *= dataloader_len
        cfg["max_epoch"] *= dataloader_len
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **cfg)
    elif scheduler_type == "LinearWarmupMultiStepLR":
        cfg.pop("max_epoch")
        cfg["warmup_epoch"] *= dataloader_len
        cfg["milestones"] = [dataloader_len * step for step in cfg["milestones"]]
        scheduler = LinearWarmupMultiStepLR(optimizer, **cfg)
    elif scheduler_type == "MultiStepLR":
        cfg.pop("max_epoch")
        cfg["milestones"] = [dataloader_len * step for step in cfg["milestones"]]
        scheduler = LinearWarmupMultiStepLR(optimizer, warmup_epoch=0, **cfg)
    else:
        raise f"Optimizer {scheduler_type} is not supported so far."

    return scheduler, max_epoch


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epoch=10, max_epoch=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer,
        warmup_epoch,
        max_epoch,
        warmup_start_lr=0.0,
        eta_min=1e-8,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epoch (int): Maximum number of iterations for linear warmup
            max_epoch (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epoch:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epoch - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epoch:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epoch) % (2 * (self.max_epoch - self.warmup_epoch)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epoch - self.warmup_epoch))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch)))
            / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch - 1) / (self.max_epoch - self.warmup_epoch)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epoch:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epoch - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch)))
            for base_lr in self.base_lrs
        ]


class LinearWarmupMultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        last_epoch=-1,
        warmup_epoch=0,
        warmup_start_lr=0,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr
        super(LinearWarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        # last epoch actually means last iter
        if self.last_epoch == self.warmup_epoch:
            return self.base_lrs
        elif self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epoch:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / self.warmup_epoch
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch in self.milestones:
            return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch) for base_lr in self.base_lrs]
