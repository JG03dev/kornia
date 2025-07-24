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

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor

INF = 1e9


def mask_border(m: Tensor, b: int, v: Union[Tensor, float, bool]) -> None:
    """Mask borders with value.

    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int): border size.
        v (m.dtype): border value.
    """
    if b <= 0:
        return
    # Apply border mask in a vectorized manner for all border slices simultaneously
    slices = (slice(None),)
    m[slices + (slice(0, b),)] = v  # H0 leading
    m[slices + (slice(None), slice(0, b))] = v  # W0 leading
    m[slices + (slice(None), slice(None), slice(0, b))] = v  # H1 leading
    m[slices + (slice(None), slice(None), slice(None), slice(0, b))] = v  # W1 leading
    m[slices + (slice(-b, None),)] = v  # H0 trailing
    m[slices + (slice(None), slice(-b, None))] = v  # W0 trailing
    m[slices + (slice(None), slice(None), slice(-b, None))] = v  # H1 trailing
    m[slices + (slice(None), slice(None), slice(None), slice(-b, None))] = v  # W1 trailing


def mask_border_with_padding(m: Tensor, bd: int, v: Union[Tensor, float, bool], p_m0: Tensor, p_m1: Tensor) -> None:
    """Apply masking to a padded boarder."""
    if bd <= 0:
        return

    N = m.shape[0]
    # Border mask (same as mask_border for the leading borders)
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    # Compute per-batch slice indices efficiently (avoid Python-level for loop if possible)
    # Each gets [N, ...] shape, .sum along dims is [N, ...]
    # After int(), these are 1D tensors of shape [N] each
    h0s = p_m0.sum(1).max(-1)[0].to(torch.int64)
    w0s = p_m0.sum(-1).max(-1)[0].to(torch.int64)
    h1s = p_m1.sum(1).max(-1)[0].to(torch.int64)
    w1s = p_m1.sum(-1).max(-1)[0].to(torch.int64)

    # Only iterate for N (should be much smaller than full tensor size)
    for b_idx in range(N):
        h0, w0, h1, w1 = h0s[b_idx], w0s[b_idx], h1s[b_idx], w1s[b_idx]
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def compute_max_candidates(p_m0: Tensor, p_m1: Tensor) -> Tensor:
    """Compute the max candidates of all pairs within a batch.

    Args:
        p_m0: padded mask 0
        p_m1: padded mask 1

    """
    h0s = p_m0.sum(1).max(-1)[0]
    w0s = p_m0.sum(-1).max(-1)[0]
    h1s = p_m1.sum(1).max(-1)[0]
    w1s = p_m1.sum(-1).max(-1)[0]
    # Don't create an intermediate stack if not needed - just multiply then min
    area0 = h0s * w0s
    area1 = h1s * w1s
    max_cand = torch.sum(torch.minimum(area0, area1))
    return max_cand


class CoarseMatching(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        # general config
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        # -- # for training fine-level LoFTR
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

        # we provide 2 options for differentiable matching
        self.match_type = config["match_type"]
        if self.match_type == "dual_softmax":
            self.temperature = config["dsmax_temperature"]
        elif self.match_type == "sinkhorn":
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!") from None
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(torch.tensor(config["skh_init_bin_score"], requires_grad=True))
            self.skh_iters = config["skh_iters"]
            self.skh_prefilter = config["skh_prefilter"]
        else:
            raise NotImplementedError

    def forward(
        self,
        feat_c0: Tensor,
        feat_c1: Tensor,
        data: Dict[str, Tensor],
        mask_c0: Optional[Tensor] = None,
        mask_c1: Optional[Tensor] = None,
    ) -> None:
        """Run forward.

        Args:
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.

        """
        _, L, S, _ = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = (feat / feat.shape[-1] ** 0.5 for feat in [feat_c0, feat_c1])

        if self.match_type == "dual_softmax":
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            if mask_c0 is not None and mask_c1 is not None:
                sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == "sinkhorn":
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None and mask_c1 is not None:
                sim_matrix[:, :L, :S].masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config["sparse_spvs"]:
                data.update({"conf_matrix_with_bin": assign_matrix.clone()})

        data.update({"conf_matrix": conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix: Tensor, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Get corase matching.

        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']

        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}

        """
        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        _device = conf_matrix.device
        thr = self.thr

        # 1. confidence thresholding
        N = conf_matrix.shape[0]
        h0c, w0c, h1c, w1c = (
            axes_lengths["h0c"],
            axes_lengths["w0c"],
            axes_lengths["h1c"],
            axes_lengths["w1c"],
        )
        mask = conf_matrix > thr
        mask = mask.reshape(N, h0c, w0c, h1c, w1c)

        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])
        # Rearrange to 3D mask: [N, h0c*w0c, h1c*w1c]
        mask = mask.reshape(N, h0c * w0c, h1c * w1c)

        # 2. mutual nearest
        # Using broadcasting and in-place logic to speed up
        conf_max2 = conf_matrix.max(dim=2, keepdim=True)[0]
        conf_max1 = conf_matrix.max(dim=1, keepdim=True)[0]
        mask = mask & (conf_matrix == conf_max2) & (conf_matrix == conf_max1)

        # 3. find all valid coarse matches
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            if self.train_pad_num_gt_min >= num_matches_train:
                raise ValueError("min-num-gt-pad should be less than num-train-matches")
            pred_limit = num_matches_train - self.train_pad_num_gt_min

            if num_matches_pred <= pred_limit:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(num_matches_pred, (pred_limit,), device=_device)

            gt_pad_size = max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min)
            gt_pad_indices = torch.randint(len(data["spv_b_ids"]), (gt_pad_size,), device=_device)
            mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=_device)

            b_ids, i_ids, j_ids, mconf = (
                torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0)
                for x, y in zip(
                    [b_ids, data["spv_b_ids"]],
                    [i_ids, data["spv_i_ids"]],
                    [j_ids, data["spv_j_ids"]],
                    [mconf, mconf_gt],
                )
            )

        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # 4. Update with matches in original image resolution
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale

        mkpts0_c = torch.stack([i_ids % w0c, i_ids // w0c], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % w1c, j_ids // w1c], dim=1) * scale1

        match_mask = mconf != 0
        gt_mask = mconf == 0

        coarse_matches.update(
            {
                "gt_mask": gt_mask,
                "m_bids": b_ids[match_mask],
                "mkpts0_c": mkpts0_c[match_mask].to(dtype=conf_matrix.dtype),
                "mkpts1_c": mkpts1_c[match_mask].to(dtype=conf_matrix.dtype),
                "mconf": mconf[match_mask],
            }
        )
        return coarse_matches
