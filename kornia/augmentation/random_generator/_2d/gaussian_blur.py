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

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from kornia.augmentation.random_generator.base import (
    RandomGeneratorBase,
    UniformDistribution,
)
from kornia.augmentation.utils import (
    _joint_range_check,
)
from kornia.core import Tensor

__all__ = ["RandomGaussianBlurGenerator"]


class RandomGaussianBlurGenerator(RandomGeneratorBase):
    r"""Generate random gaussian blur parameters for a batch of images.

    Args:
        sigma: The range to uniformly sample the standard deviation for the Gaussian kernel.

    Returns:
        A dict of parameters to be passed for transformation.
            - sigma: element-wise standard deviation with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, sigma: Union[Tuple[float, float], Tensor] = (0.1, 2.0)) -> None:
        super().__init__()
        if sigma[1] < sigma[0]:
            raise TypeError(f"sigma_max should be higher than sigma_min: {sigma} passed.")

        self.sigma = sigma
        self.sigma_sampler: UniformDistribution

    def __repr__(self) -> str:
        repr_buf = f"sigma={self.sigma}"
        return repr_buf

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if not isinstance(self.sigma, (torch.Tensor)):
            sigma = torch.tensor(self.sigma, device=device, dtype=dtype)
        else:
            sigma = self.sigma.to(device=device, dtype=dtype)

        _joint_range_check(sigma, "sigma", (0, float("inf")))

        self.sigma_sampler = UniformDistribution(sigma[0], sigma[1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        # Inlined fast parameter check
        batch_size = batch_shape[0]
        _optimized_common_param_check(batch_size, same_on_batch)
        # Use faster sampling variant
        sigma = _fast_adapted_rsampling((batch_size,), self.sigma_sampler, same_on_batch)
        return {"sigma": sigma}


# Inlined and optimized _common_param_check for speed
def _optimized_common_param_check(batch_size: int, same_on_batch: Optional[bool] = None) -> None:
    # Assume fast-path for valid cases, short-circuit errors
    if type(batch_size) is not int or batch_size < 0:
        raise AssertionError(f"`batch_size` shall be a positive integer. Got {batch_size}.")
    if same_on_batch is not None and type(same_on_batch) is not bool:
        raise AssertionError(f"`same_on_batch` shall be boolean. Got {same_on_batch}.")


# Optimized adapted_rsampling: avoid redundant operation, micro-optimize for shape construction
def _fast_adapted_rsampling(
    shape: Union[Tuple[int, ...], torch.Size],
    dist: torch.distributions.Distribution,
    same_on_batch: Optional[bool] = False,
) -> Tensor:
    # Remove isinstance() if already size, shortcut TORCH special-casing
    if not isinstance(shape, torch.Size):
        # This branch infrequent, so avoid assignment if possible
        shape = torch.Size(shape)

    if same_on_batch:
        B = shape[0]
        # (1, ...) sample: don't build new tuple if single dim
        rsample_size = (1,) + tuple(shape[1:])
        # sample once and expand efficiently, avoids extra repeat of size-one dims for 1D
        rsample = dist.rsample(rsample_size)
        # Use expand where possible for efficiency
        out = rsample.expand(shape)
        return out
    else:
        return dist.rsample(shape)
