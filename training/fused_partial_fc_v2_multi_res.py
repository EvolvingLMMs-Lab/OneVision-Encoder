import math
import os
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones(
                    [index_positive.size(0), logits.size(1)], device=logits.device
                )
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.s == 1:
            return logits

        if self.m1 == 1.0 and self.m3 == 0.0:
            # sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            # cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            # if self.easy_margin:
            #     final_target_logit = torch.where(
            #         target_logit > 0, cos_theta_m, target_logit)
            # else:
            #     final_target_logit = torch.where(
            #         target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            # logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            # logits = logits * self.s
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = (
                    final_target_logit
                )
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """

    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
        is_normlize: int = 1,
        sample_num_feat=None,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy_multi_fused = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.sample_num_feat: int = sample_num_feat
        self.fp16 = fp16
        self.is_normlize = is_normlize
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_local, embedding_size))
        )

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def sample(self, labels, index_positive):
        """
        This functions will change the value of labels
        Parameters:
        -----------
        labels: torch.Tensor
            pass
        index_positive: torch.Tensor
            pass
        optimizer: torch.optim.Optimizer
            pass
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).cuda()
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                index = index.sort()[0].cuda()
            else:
                index = positive
            # self.weight_index = index
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        return self.weight[index], labels
        # return index, labels

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        random_diff,
    ):
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)

        with torch.no_grad():
            noise = torch.rand(batch_size, random_diff, device="cuda")
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :3]
            local_labels = torch.gather(local_labels, dim=1, index=ids_keep)

        ########################### added by multi-res ###########################
        # Gather sizes of each feature tensor
        batch_size_pt = torch.tensor([batch_size], device="cuda")
        gathered_batch_size = [torch.zeros_like(batch_size_pt) for _ in range(self.world_size)]
        distributed.all_gather(gathered_batch_size, batch_size_pt)
        gathered_batch_size = [size.item() for size in gathered_batch_size]    
        # max_batch_size = max(gathered_batch_size)
        distributed.all_reduce(batch_size_pt, distributed.ReduceOp.MAX)
        max_batch_size = batch_size_pt.item()
        # Pad features to the maximum size
        if local_embeddings.size(0) < max_batch_size:
            padding = (0, 0, 0, int(max_batch_size - local_embeddings.size(0)))
            local_embeddings = torch.nn.functional.pad(local_embeddings, padding)

        if local_labels.size(0) < max_batch_size:
            padding = (0, 0, 0, int(max_batch_size - local_labels.size(0)))
            local_labels = torch.nn.functional.pad(local_labels, padding)
        ########################### added by multi-res ###########################

        _gather_embeddings = [
            torch.zeros_like(local_embeddings)
            for _ in range(self.world_size)
        ]

        # print(local_embeddings.size())
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)

        # print(local_labels.size())
        _gather_labels = [
            torch.zeros_like(local_labels) for _ in range(self.world_size)
        ]
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        ########################### added by multi-res ###########################
        # Remove padding
        _all_embeddings = [feat[:size] for feat, size in zip(embeddings.split(max_batch_size), gathered_batch_size)]
        _all_labels = [feat[:size] for feat, size in zip(labels.split(max_batch_size), gathered_batch_size)]
        
        embeddings = torch.cat(_all_embeddings, dim=0)
        labels = torch.cat(_all_labels, dim=0)
        ########################### added by multi-res ###########################


        total_batch_size = labels.size(0)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        labels = labels.reshape(total_batch_size, 3)
        index_positive = index_positive.reshape(total_batch_size, 3)

        labels_0 = labels[:, 0:1].clone()
        labels_1 = labels[:, 1:2].clone()
        labels_2 = labels[:, 2:3].clone()

        index_positive_0 = index_positive[:, 0:1].clone()
        index_positive_1 = index_positive[:, 1:2].clone()
        index_positive_2 = index_positive[:, 2:3].clone()

        weight_0, labels_0 = self.sample(labels_0, index_positive_0)
        weight_1, labels_1 = self.sample(labels_1, index_positive_1)
        weight_2, labels_2 = self.sample(labels_2, index_positive_2)

        with torch.cuda.amp.autocast(self.fp16):
            if self.is_normlize:
                norm_embeddings = normalize(embeddings)
                norm_weight_activated_0 = normalize(weight_0)
                norm_weight_activated_1 = normalize(weight_1)
                norm_weight_activated_2 = normalize(weight_2)
                logits_0 = linear(norm_embeddings, norm_weight_activated_0)
                logits_1 = linear(norm_embeddings, norm_weight_activated_1)
                logits_2 = linear(norm_embeddings, norm_weight_activated_2)
            else:
                raise NotImplementedError

        logits_0 = logits_0.float()
        logits_1 = logits_1.float()
        logits_2 = logits_2.float()

        if self.is_normlize:
            logits_0 = logits_0.clamp(-1, 1)
            logits_1 = logits_1.clamp(-1, 1)
            logits_2 = logits_2.clamp(-1, 1)
        else:
            raise NotImplementedError

        logits_0 = self.margin_softmax(logits_0, labels_0)
        logits_1 = self.margin_softmax(logits_1, labels_1)
        logits_2 = self.margin_softmax(logits_2, labels_2)

        loss = self.dist_cross_entropy_multi_fused(
            logits_0, logits_1, logits_2, labels_0, labels_1, labels_2
        )
        return loss


class FusedDistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(
        ctx,
        logits_0: torch.Tensor,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        label_0: torch.Tensor,
        label_1: torch.Tensor,
        label_2: torch.Tensor,
    ):
        """ """
        index_positive_0 = torch.where(label_0 != -1)[0]
        index_positive_1 = torch.where(label_1 != -1)[0]
        index_positive_2 = torch.where(label_2 != -1)[0]

        batch_size = logits_0.size(0)
        # for numerical stability
        max_logits_0, _ = torch.max(logits_0, dim=1, keepdim=True)
        max_logits_1, _ = torch.max(logits_1, dim=1, keepdim=True)
        max_logits_2, _ = torch.max(logits_2, dim=1, keepdim=True)

        chunk_max_logits = torch.cat([max_logits_0, max_logits_1, max_logits_2], dim=0)
        # local to global
        distributed.all_reduce(chunk_max_logits, distributed.ReduceOp.MAX)
        # print(len(torch.split(chunk_max_logits, max_logits_0.size(0), dim=0)))
        max_logits_0, max_logits_1, max_logits_2 = torch.split(
            chunk_max_logits, max_logits_0.size(0), dim=0
        )

        logits_0.sub_(max_logits_0).exp_()
        logits_1.sub_(max_logits_1).exp_()
        logits_2.sub_(max_logits_2).exp_()
        # logits.sub_(max_logits)
        # logits.exp_()

        sum_logits_exp_0 = torch.sum(logits_0, dim=1, keepdim=True)
        sum_logits_exp_1 = torch.sum(logits_1, dim=1, keepdim=True)
        sum_logits_exp_2 = torch.sum(logits_2, dim=1, keepdim=True)

        chunk_sum_logits_exp = torch.cat(
            [sum_logits_exp_0, sum_logits_exp_1, sum_logits_exp_2], dim=0
        )
        distributed.all_reduce(chunk_sum_logits_exp, distributed.ReduceOp.SUM)
        sum_logits_exp_0, sum_logits_exp_1, sum_logits_exp_2 = torch.split(
            chunk_sum_logits_exp, sum_logits_exp_0.size(0), dim=0
        )

        # local to global
        # distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)

        logits_0.div_(sum_logits_exp_0)
        logits_1.div_(sum_logits_exp_1)
        logits_2.div_(sum_logits_exp_2)

        # logits.div_(sum_logits_exp)
        # index = torch.where(label != -1)[0]
        # loss
        loss_0 = torch.zeros(batch_size, 1, device="cuda")
        loss_1 = torch.zeros(batch_size, 1, device="cuda")
        loss_2 = torch.zeros(batch_size, 1, device="cuda")

        loss_0[index_positive_0] = logits_0[index_positive_0].gather(
            1, label_0[index_positive_0]
        )
        loss_1[index_positive_1] = logits_1[index_positive_1].gather(
            1, label_1[index_positive_1]
        )
        loss_2[index_positive_2] = logits_2[index_positive_2].gather(
            1, label_2[index_positive_2]
        )

        chunk_loss = torch.cat([loss_0, loss_1, loss_2], dim=0)
        distributed.all_reduce(chunk_loss, distributed.ReduceOp.SUM)
        loss_0, loss_1, loss_2 = torch.split(chunk_loss, loss_0.size(0), dim=0)

        loss_0 = loss_0.clamp_min_(1e-30).log_().mean() * (-1)
        loss_1 = loss_1.clamp_min_(1e-30).log_().mean() * (-1)
        loss_2 = loss_2.clamp_min_(1e-30).log_().mean() * (-1)
        loss = (loss_0 + loss_1 + loss_2) / 3
        # loss[index] = logits[index].gather(1, label[index])
        # distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(
            index_positive_0,
            index_positive_1,
            index_positive_2,
            logits_0,
            logits_1,
            logits_2,
            label_0,
            label_1,
            label_2,
        )
        # return loss.clamp_min_(1e-30).log_().mean() * (-1)
        return loss

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index_positive_0,
            index_positive_1,
            index_positive_2,
            logits_0,
            logits_1,
            logits_2,
            label_0,
            label_1,
            label_2,
        ) = ctx.saved_tensors

        batch_size = logits_0.size(0)
        one_hot_0 = torch.zeros(
            size=[index_positive_0.size(0), logits_0.size(1)], device="cuda"
        )
        one_hot_1 = torch.zeros(
            size=[index_positive_1.size(0), logits_1.size(1)], device="cuda"
        )
        one_hot_2 = torch.zeros(
            size=[index_positive_2.size(0), logits_2.size(1)], device="cuda"
        )
        one_hot_0.scatter_(1, label_0[index_positive_0], 1)
        one_hot_1.scatter_(1, label_1[index_positive_1], 1)
        one_hot_2.scatter_(1, label_2[index_positive_2], 1)

        logits_0[index_positive_0] -= one_hot_0
        logits_1[index_positive_1] -= one_hot_1
        logits_2[index_positive_2] -= one_hot_2
        logits_0.div_(batch_size)
        logits_1.div_(batch_size)
        logits_2.div_(batch_size)

        # one_hot.scatter_(1, label[index], 1)
        # logits[index] -= one_hot
        # logits.div_(batch_size)
        # return logits * loss_gradient.item(), None
        output = (
            logits_0 * loss_gradient.item(),
            logits_1 * loss_gradient.item(),
            logits_2 * loss_gradient.item(),
            None,
            None,
            None,
        )
        return output


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(
        self,
        logits_0: torch.Tensor,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        label_0: torch.Tensor,
        label_1: torch.Tensor,
        label_2: torch.Tensor,
    ):
        return FusedDistCrossEntropyFunc.apply(
            logits_0,
            logits_1,
            logits_2,
            label_0,
            label_1,
            label_2,
        )


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            (
                distributed.reduce(
                    grad_out, rank, distributed.ReduceOp.SUM, async_op=True
                )
                if i == rank
                else distributed.reduce(
                    grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
                )
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
