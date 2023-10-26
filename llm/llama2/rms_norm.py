# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn, Tensor


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Attributes:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6

    Args:
        x (Tensor): input tensor to normalize

    Returns:
        torch.Tensor: The output tensor after applying RMSNorm.

    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # computation is in fp32
        x = x.float()
        x_normed = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale
        ).type_as(x)
        return x_normed * self.scale
