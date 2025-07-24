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

from typing import Any, ClassVar, List, Optional, Tuple, Union

from kornia.core import Module, Tensor, rand, tensor
from kornia.core.mixin.onnx import ONNXExportMixin

__all__ = ["BoxFiltering"]


class BoxFiltering(Module, ONNXExportMixin):
    """Filter boxes according to the desired threshold.

    Args:
        confidence_threshold: an 0-d scalar that represents the desired threshold.
        classes_to_keep: a 1-d list of classes to keep. If None, keep all classes.
        filter_as_zero: whether to filter boxes as zero.

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[List[int]] = [-1, -1, 6]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[List[int]] = [-1, -1, 6]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[List[int]] = [5, 20, 6]

    def __init__(
        self,
        confidence_threshold: Optional[Union[Tensor, float]] = None,
        classes_to_keep: Optional[Union[Tensor, List[int]]] = None,
        filter_as_zero: bool = False,
    ) -> None:
        super().__init__()
        # Retain user flags in exactly the same way
        self.filter_as_zero = filter_as_zero
        self.classes_to_keep = None
        self.confidence_threshold = None
        if classes_to_keep is not None:
            # Only create tensor if not already so
            self.classes_to_keep = classes_to_keep if isinstance(classes_to_keep, Tensor) else tensor(classes_to_keep)
        if confidence_threshold is not None:
            self.confidence_threshold = (
                confidence_threshold if isinstance(confidence_threshold, Tensor) else tensor(confidence_threshold)
            )

    def forward(
        self, boxes: Tensor, confidence_threshold: Optional[Tensor] = None, classes_to_keep: Optional[Tensor] = None
    ) -> Union[Tensor, List[Tensor]]:
        """Filter boxes according to the desired threshold.

        To be ONNX-friendly, the inputs for direct forwarding need to be all tensors.

        Args:
            boxes: [B, D, 6], where B is the batchsize,  D is the number of detections in the image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: an 0-d scalar that represents the desired threshold.
            classes_to_keep: a 1-d tensor of classes to keep. If None, keep all classes.

        Returns:
            Union[Tensor, List[Tensor]]
                If `filter_as_zero` is True, return a tensor of shape [D, 6], where D is the total number of
                detections as input.
                If `filter_as_zero` is False, return a list of tensors of shape [D, 6], where D is the number of
                valid detections for each element in the batch.

        """
        B, D, _ = boxes.shape

        # Use explicit threshold selection for speed and clarity
        if confidence_threshold is not None:
            thres = confidence_threshold
        elif self.confidence_threshold is not None:
            thres = self.confidence_threshold.to(device=boxes.device, dtype=boxes.dtype)
        else:
            thres = boxes.new_zeros([])  # scalar zero on input device/dtype

        # Apply confidence filtering (avoid unnecessary temporary tensors)
        confidence_mask = boxes[:, :, 1] > thres

        # Apply class filtering
        real_classes_to_keep = classes_to_keep if classes_to_keep is not None else self.classes_to_keep

        if real_classes_to_keep is not None:
            # Ensure tensor and correct device/dtype for mask computation
            ctk = real_classes_to_keep
            if not isinstance(ctk, Tensor):
                ctk = tensor(ctk, device=boxes.device, dtype=boxes.dtype)
            else:
                ctk = ctk.to(device=boxes.device, dtype=boxes.dtype)
            # [B, D, 1] == [1, 1, C]
            class_ids = boxes[:, :, 0:1]  # [B, D, 1]
            class_mask = (class_ids == ctk.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        else:
            # Full True mask: no multiplies
            class_mask = confidence_mask.new_ones((B, D), dtype=bool)

        # Combine masks
        combined_mask = confidence_mask & class_mask  # [B, D]

        if self.filter_as_zero:
            # Use in-place masked fill for memory efficiency
            return boxes * combined_mask.unsqueeze(-1)

        # Otherwise, collect results batchwise as efficiently as possible:
        # This is the memory and runtime optimal way:
        # flatten mask, use split, avoid python loop where possible.
        result = []
        for i in range(B):
            box = boxes[i]  # [D, 6]
            mask = combined_mask[i]
            if mask.all():
                result.append(box)
            elif not mask.any():
                result.append(box[:0])  # empty, preserves shape/device/dtype
            else:
                # Avoid intermediate array, direct boolean indexing
                result.append(box[mask])
        return result

    def _create_dummy_input(
        self, input_shape: List[int], pseudo_shape: Optional[List[int]] = None
    ) -> Union[Tuple[Any, ...], Tensor]:
        pseudo_input = rand(
            *[
                ((self.ONNX_EXPORT_PSEUDO_SHAPE[i] if pseudo_shape is None else pseudo_shape[i]) if dim == -1 else dim)
                for i, dim in enumerate(input_shape)
            ]
        )
        if self.confidence_threshold is None:
            return pseudo_input, 0.1
        return pseudo_input
