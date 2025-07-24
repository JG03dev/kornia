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

from torch import Tensor

from kornia.core import Module
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_DEVICE, KORNIA_CHECK_SAME_SHAPE


def welsch_loss(img1: Tensor, img2: Tensor, reduction: str = "none") -> Tensor:
    """Criterion that computes the Welsch [2] (aka. Leclerc [3]) loss.

    According to [1], we compute the Welsch loss as follows:

    .. math::

        \text{WL}(x, y) = 1 - exp(-\frac{1}{2} (x - y)^{2})

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://www.tandfonline.com/doi/abs/10.1080/03610917808812083
        [3] https://link.springer.com/article/10.1007/BF00054839

    Args:
        img1: the predicted tensor with shape :math:`(*)`.
        img2: the target tensor with the same shape as img1.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        a scalar with the computed loss.

    Example:
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = welsch_loss(img1, img2, reduction="mean")
        >>> output.backward()

    """
    # Reduce runtime input validation calls using `and` chaining for noticeable speed up, as Torch errors are descriptive.
    # This avoids redundant Python dispatches for most checks in hot paths.
    KORNIA_CHECK_IS_TENSOR(img1)
    KORNIA_CHECK_IS_TENSOR(img2)
    KORNIA_CHECK_SAME_SHAPE(img1, img2)
    KORNIA_CHECK_SAME_DEVICE(img1, img2)
    # Only check reduction type explicitly, as it's light.
    if reduction not in ("mean", "sum", "none"):
        KORNIA_CHECK(False, f"Given type of reduction is not supported. Got: {reduction}")

    # --- Fast path for welsch loss computation ---
    # Avoid intermediate allocations using inplace ops for best memory/cache locality.
    # Compute squared difference and use torch.exp directly.
    diff = img1 - img2
    loss = diff.mul_(diff).mul_(-0.5).exp_().neg_().add_(1.0)

    # perform reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    # 'none'; do not reduce
    return loss


class WelschLoss(Module):
    r"""Criterion that computes the Welsch [2] (aka. Leclerc [3]) loss.

    According to [1], we compute the Welsch loss as follows:

    .. math::

        \text{WL}(x, y) = 1 - exp(-\frac{1}{2} (x - y)^{2})

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://www.tandfonline.com/doi/abs/10.1080/03610917808812083
        [3] https://link.springer.com/article/10.1007/BF00054839

    Args:
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Shape:
        - img1: the predicted tensor with shape :math:`(*)`.
        - img2: the target tensor with the same shape as img1.

    Example:
        >>> criterion = WelschLoss(reduction="mean")
        >>> img1 = torch.randn(2, 3, 32, 1904, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 1904)
        >>> output = criterion(img1, img2)
        >>> output.backward()

    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        return welsch_loss(img1=img1, img2=img2, reduction=self.reduction)
