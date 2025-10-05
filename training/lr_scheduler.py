from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import SGD
import torch
import warnings


class PolynomialLRWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iters,
        total_iters=5,
        power=1.0,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            l = self.last_epoch
            w = self.warmup_iters
            t = self.total_iters
            decay_factor = (
                (1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))
            ) ** self.power
        return [(1/2 * group["lr"]) * (1+decay_factor) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            return [
                (
                    base_lr
                    * (
                        1.0
                        - (min(self.total_iters, self.last_epoch) - self.warmup_iters)
                        / (self.total_iters - self.warmup_iters)
                    )
                    ** self.power
                )
                for base_lr in self.base_lrs
            ]

    def load_state_dict(self, state_dict):
        """Override to update the optimizer's learning rates immediately after loading."""
        super().load_state_dict(state_dict)
        
        # Calculate and apply the learning rates based on the current state
        for param_group, lr in zip(self.optimizer.param_groups, self._get_closed_form_lr()):
            param_group['lr'] = lr
