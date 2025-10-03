import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# LayerNorm modules (AMP friendly)
# --------------------------------------------------------
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


# --------------------------------------------------------
# Distillation Head with MLP Projector
# --------------------------------------------------------
class MaskFC_V1(nn.Module):
    def __init__(self, teacher_dim=1024, student_dim=384, hidden_dim=512, loss_type="smoothl1"):
        """
        Args:
            teacher_dim (int): hidden dim of teacher features
            student_dim (int): hidden dim of student features
            hidden_dim (int): hidden size of the projector MLP
            loss_type (str): loss function type ["mse", "smoothl1"]
        """
        super(MaskFC_V1, self).__init__()

        # student → teacher 对齐 projector (MLP)
        self.proj_student = nn.Sequential(
            # LayerNorm(student_dim),
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )

        # teacher 特征的归一化
        # self.norm_teacher = nn.LayerNorm(teacher_dim, elementwise_affine=False)

        # 损失函数
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "smoothl1":
            self.loss_fn = nn.SmoothL1Loss(beta=2.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, student_embeddings, teacher_embeddings):
        """
        Args:
            student_embeddings (Tensor): [B, N, student_dim] or [B, student_dim]
            teacher_embeddings (Tensor): [B, N, teacher_dim] or [B, teacher_dim]
        """
        # 投影 student
        student_proj = self.proj_student(student_embeddings)

        # 归一化 teacher
        # teacher_proj = self.norm_teacher(teacher_embeddings)

        # 计算 loss
        loss = self.loss_fn(student_proj, teacher_embeddings)
        return loss