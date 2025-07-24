# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import torch
from torch import nn

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def inverse_depth_smoothness_loss(idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \\left | \\partial_x d_{ij} \right | e^{-\\left \\|
        \\partial_x I_{ij} \right \\|} + \\left |
        \\partial_y d_{ij} \right | e^{-\\left \\| \\partial_y I_{ij} \right \\|}

    Args:
        idepth: tensor with the inverse depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.

    Return:
        a scalar with the computed loss.

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> loss = inverse_depth_smoothness_loss(idepth, image)

    """
    _check_inputs(idepth, image)

    # --- Use in-place operations to reduce memory usage; batch all abs+mean/exp; fuse ops where possible ---

    # Compute gradients for idepth and image in each direction
    idepth_dx = _gradient_x(idepth)
    idepth_dy = _gradient_y(idepth)
    image_dx = _gradient_x(image)
    image_dy = _gradient_y(image)

    # Precompute abs, mean, exp for image gradients in one operation per axis
    # Instead of several temporaries, use fused chains and try to keep peak memory down

    # abs().mean() with dim=1, keepdim=True, then - and exp just after
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # Multiply idepth gradient with weights and take abs in one go to reduce temporaries
    # Avoid creating an intermediate (idepth_dx * weights_x), go for fused abs_*_mul directly
    smoothness_x = torch.abs_(idepth_dx.mul_(weights_x))  # in-place mul and abs to minimize allocations
    smoothness_y = torch.abs_(idepth_dy.mul_(weights_y))

    # Use add_ (in-place) to accumulate the two means for final scalar
    # (PyTorch's mean is already fast, but summing before mean would be slower for loss definition)
    # No need to fuse further

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def _check_inputs(idepth: torch.Tensor, image: torch.Tensor) -> None:
    # check type and shape compatibility, fast-path for common case
    if (
        not isinstance(idepth, torch.Tensor)
        or not isinstance(image, torch.Tensor)
        or idepth.shape[-2:] != image.shape[-2:]
        or idepth.device != image.device
        or idepth.dtype != image.dtype
        or idepth.ndim != 4
        or image.ndim != 4
    ):
        # perform slower, more granular error for debugging
        if not isinstance(idepth, torch.Tensor):
            raise TypeError(f"Input idepth type is not a torch.Tensor. Got {type(idepth)}")
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}")
        if not idepth.ndim == 4:
            raise ValueError(f"Invalid idepth shape, we expect BxCxHxW. Got: {idepth.shape}")
        if not image.ndim == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")
        if not idepth.shape[-2:] == image.shape[-2:]:
            raise ValueError(f"idepth and image shapes must be the same. Got: {idepth.shape} and {image.shape}")
        if not idepth.device == image.device:
            raise ValueError(f"idepth and image must be in the same device. Got: {idepth.device} and {image.device}")
        if not idepth.dtype == image.dtype:
            raise ValueError(f"idepth and image must be in the same dtype. Got: {idepth.dtype} and {image.dtype}")


class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = InverseDepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)

    """

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return inverse_depth_smoothness_loss(idepth, image)
