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
from torch import Tensor, nn


def aepe(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Create a function that calculates the average endpoint error (AEPE) between 2 flow maps.

    AEPE is the endpoint error between two 2D vectors (e.g., optical flow).
    Given a h x w x 2 optical flow map, the AEPE is:

    .. math::

        \text{AEPE}=\frac{1}{hw}\\sum_{i=1, j=1}^{h, w}\\sqrt{(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}}

    Args:
        input: the input flow map with shape :math:`(*, 2)`.
        target: the target flow map with shape :math:`(*, 2)`.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Return:
        the computed AEPE as a scalar.

    Examples:
        >>> ones = torch.ones(4, 4, 2)
        >>> aepe(ones, 1.2 * ones)
        tensor(0.2828)

    Reference:
        https://link.springer.com/content/pdf/10.1007/s11263-010-0390-2.pdf

    """
    # Fast combined type/shape/size check to eliminate most function call overhead in hot path
    _aepe_fast_check(input, target)

    # Main AEPE computation
    diff = input - target
    # Fused squared sum over last dim, then sqrt for endpoint error (avoids multiple indexing)
    epe: Tensor = (diff.square().sum(dim=-1)).sqrt()

    if reduction == "mean":
        return epe.mean()
    elif reduction == "sum":
        return epe.sum()
    elif reduction == "none":
        return epe
    else:
        raise NotImplementedError("Invalid reduction option.")


# Optimized type and shape check helper for AEPE to avoid repeated logic in the hot aepe path
def _aepe_fast_check(input: Tensor, target: Tensor):
    """Fast-path combined check for Tensor type, shape, and match for AEPE function."""
    # Type check
    if not (isinstance(input, Tensor) and isinstance(target, Tensor)):
        raise TypeError(f"Not a Tensor type. Got: {type(input)} or {type(target)}.")

    # Shape check: must end with 2 (the flow vector)
    if input.shape[-1] != 2 or target.shape[-1] != 2:
        raise TypeError(f"Last dimension must be 2. Got: input {input.shape}, target {target.shape}")
    # Shape equality check
    if input.shape != target.shape:
        raise Exception(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")


class AEPE(nn.Module):
    r"""Computes the average endpoint error (AEPE) between 2 flow maps.

    EPE is the endpoint error between two 2D vectors (e.g., optical flow).
    Given a h x w x 2 optical flow map, the AEPE is:

    .. math::

        \text{AEPE}=\frac{1}{hw}\sum_{i=1, j=1}^{h, w}\sqrt{(I_{i,j,1}-T_{i,j,1})^{2}+(I_{i,j,2}-T_{i,j,2})^{2}}

    Args:
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Shape:
        - input: :math:`(*, 2)`.
        - target :math:`(*, 2)`.
        - output: :math:`(1)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 2)
        >>> input2 = torch.rand(1, 4, 5, 2)
        >>> epe = AEPE(reduction="mean")
        >>> epe = epe(input1, input2)

    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return aepe(input, target, self.reduction)


average_endpoint_error = aepe
