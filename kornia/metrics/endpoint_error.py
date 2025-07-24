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
    # Fast, inlined checks (much cheaper than repeated function call overhead)
    _fast_aepe_checks(input, target)

    # Vectorized difference and squared sum for two last channels, no .sqrt broadcasting
    diff = input - target
    epe = torch.linalg.norm(diff, ord=2, dim=-1)  # Faster, more robust than manual (x**2+y**2).sqrt()

    if reduction == "mean":
        return epe.mean()
    elif reduction == "sum":
        return epe.sum()
    elif reduction == "none":
        return epe
    else:
        raise NotImplementedError("Invalid reduction option.")


def _fast_aepe_checks(input: torch.Tensor, target: torch.Tensor):
    # Combined faster checks for type, shape, and equality.
    # We inline equivalent minimal safety check logic here to drastically reduce
    # slow Python overhead from repeated function calls.
    # Only do the checks if assert statements are not optimized out.
    if __debug__:
        if not isinstance(input, Tensor):
            raise TypeError(f"Not a Tensor type. Got: {type(input)}.")
        if not isinstance(target, Tensor):
            raise TypeError(f"Not a Tensor type. Got: {type(target)}.")
        if input.shape[-1] != 2:
            raise TypeError(f"input shape must be (*, 2). Got {input.shape}")
        if target.shape[-1] != 2:
            raise TypeError(f"target shape must be (*, 2). Got {target.shape}")
        if input.shape != target.shape:
            raise Exception(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")
    # else skip entirely for optimized running


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
