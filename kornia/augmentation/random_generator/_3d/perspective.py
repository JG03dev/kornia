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

from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor, as_tensor, tensor
from kornia.utils.helpers import _extract_device_dtype


class PerspectiveGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: controls the degree of distortion and ranges from 0 to 1.

    Returns:
        A dict of parameters to be passed for transformation.
            - src (Tensor): perspective source bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): perspective target bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, distortion_scale: Union[Tensor, float] = 0.5) -> None:
        super().__init__()
        self.distortion_scale = distortion_scale

    def __repr__(self) -> str:
        repr = f"distortion_scale={self.distortion_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}")
        self.rand_sampler = Uniform(
            tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        depth = batch_shape[-3]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.distortion_scale])

        # -- OPTIMIZATION: Prepare the point grids directly as arrays and cast with torch.as_tensor on target device/dtype to avoid slow tensor() calls --

        # static -- shape (8, 3)
        _start_points_data = [
            [0.0, 0, 0],  # Front-top-left
            [width - 1, 0, 0],  # Front-top-right
            [width - 1, height - 1, 0],  # Front-bottom-right
            [0, height - 1, 0],  # Front-bottom-left
            [0.0, 0, depth - 1],  # Back-top-left
            [width - 1, 0, depth - 1],  # Back-top-right
            [width - 1, height - 1, depth - 1],  # Back-bottom-right
            [0, height - 1, depth - 1],  # Back-bottom-left
        ]
        # Static, shape: (8, 3)
        _pts_norm_data = [
            [1, 1, 1],  # Front-top-left
            [-1, 1, 1],  # Front-top-right
            [-1, -1, 1],  # Front-bottom-right
            [1, -1, 1],  # Front-bottom-left
            [1, 1, -1],  # Back-top-left
            [-1, 1, -1],  # Back-top-right
            [-1, -1, -1],  # Back-bottom-right
            [1, -1, -1],  # Back-bottom-left
        ]

        # Fast tensor allocation for start points (shape: (1, 8, 3), then expand to (B, 8, 3))
        start_points = torch.as_tensor([_start_points_data], device=_device, dtype=_dtype).expand(batch_size, -1, -1)

        # Fast tensor allocation for pts_norm (shape: (1, 8, 3)), no need to expand as broadcasting will work
        pts_norm = torch.as_tensor(
            [_pts_norm_data],
            device=_device,
            dtype=_dtype,
        )

        # Precompute the distortion factors efficiently
        if isinstance(self.distortion_scale, torch.Tensor):
            ds = self.distortion_scale.item()
        else:
            ds = self.distortion_scale

        fx = ds * width * 0.5
        fy = ds * height * 0.5
        fz = ds * depth * 0.5

        # Batch is dim0, so shape is (1, 1, 3)
        factor = torch.tensor([[fx, fy, fz]], device=_device, dtype=_dtype).view(1, 1, 3)

        # Efficient random sample generation
        shape = (batch_size, 8, 3)
        rand_val = _adapted_rsampling(shape, self.rand_sampler, same_on_batch)
        if rand_val.device != _device or rand_val.dtype != _dtype:
            rand_val = rand_val.to(device=_device, dtype=_dtype)

        # -- OPTIMIZATION: Use in-place operations to save memory and avoid unnecessary temporaries for element-wise operations --
        # pts_norm and factor will broadcast to (B, 8, 3) as needed
        # Instead of multiple multiplications, do it in one call using broadcasting
        # end_points = start_points + factor * rand_val * pts_norm
        end_points = torch.addcmul(start_points, factor * pts_norm, rand_val)  # a + b * c (elementwise)

        return {"start_points": start_points, "end_points": end_points}
