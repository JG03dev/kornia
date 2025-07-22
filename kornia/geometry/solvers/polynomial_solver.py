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

"""Module containing the functionalities for computing the real roots of polynomial equation."""

from __future__ import annotations

import math

import torch

from kornia.core import Tensor, cos, stack, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE


# Reference : https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/polynom_solver.cpp
def solve_quadratic(coeffs: Tensor) -> Tensor:
    """Solve given quadratic equation.

    The function takes the coefficients of quadratic equation and returns the real roots.

    .. math:: coeffs[0]x^2 + coeffs[1]x + coeffs[2] = 0

    Args:
        coeffs : The coefficients of quadratic equation :`(B, 3)`

    Returns:
        A tensor of shape `(B, 2)` containing the real roots to the quadratic equation.

    Example:
        >>> coeffs = torch.tensor([[1., 4., 4.]])
        >>> roots = solve_quadratic(coeffs)

    .. note::
       In cases where a quadratic polynomial has only one real root, the output will be in the format
       [real_root, 0]. And for the complex roots should be represented as 0. This is done to maintain
       a consistent output shape for all cases.

    """
    KORNIA_CHECK_SHAPE(coeffs, ["B", "3"])

    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]

    delta = b * b - 4 * a * c

    mask_negative = delta < 0
    mask_zero = delta == 0
    mask_positive = ~mask_negative & ~mask_zero  # combine to avoid recomputation

    inv_2a = 0.5 / a

    # Inplace zeros for nonreal solutions
    solutions = zeros((coeffs.shape[0], 2), device=coeffs.device, dtype=coeffs.dtype)

    if torch.any(mask_zero):
        idx = mask_zero.nonzero(as_tuple=True)[0]
        v = (-b[idx]) * inv_2a[idx]
        solutions[idx, 0] = v
        solutions[idx, 1] = v

    if torch.any(mask_positive):
        sqrt_delta = torch.sqrt(delta[mask_positive])
        idx = mask_positive.nonzero(as_tuple=True)[0]
        b_v, sqrt_v, inv_2a_v = b[idx], sqrt_delta, inv_2a[idx]
        root1 = (-b_v + sqrt_v) * inv_2a_v
        root2 = (-b_v - sqrt_v) * inv_2a_v
        solutions[idx, 0] = root1
        solutions[idx, 1] = root2

    return solutions


def solve_cubic(coeffs: Tensor) -> Tensor:
    """Solve given cubic equation.

    The function takes the coefficients of cubic equation and returns
    the real roots.

    .. math:: coeffs[0]x^3 + coeffs[1]x^2 + coeffs[2]x + coeffs[3] = 0

    Args:
        coeffs : The coefficients cubic equation : `(B, 4)`

    Returns:
        A tensor of shape `(B, 3)` containing the real roots to the cubic equation.

    Example:
        >>> coeffs = torch.tensor([[32., 3., -11., -6.]])
        >>> roots = solve_cubic(coeffs)

    .. note::
       In cases where a cubic polynomial has only one or two real roots, the output for the non-real
       roots should be represented as 0. Thus, the output for a single real root should be in the
       format [real_root, 0, 0], and for two real roots, it should be [real_root_1, real_root_2, 0].

    """
    KORNIA_CHECK_SHAPE(coeffs, ["B", "4"])
    device, dtype = coeffs.device, coeffs.dtype
    pi_val = math.pi

    # Fast constant setup
    _PI = torch.full((1,), pi_val, device=device, dtype=dtype)

    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    d = coeffs[:, 3]
    N = coeffs.shape[0]

    solutions = zeros((N, 3), device=device, dtype=dtype)

    mask_a_zero = a == 0
    mask_b_zero = b == 0
    mask_c_zero = c == 0

    # Linear degenerate case, only fill those (solutions already 0 elsewhere)
    mask_first_order = mask_a_zero & mask_b_zero & ~mask_c_zero
    if torch.any(mask_first_order):
        idx = mask_first_order.nonzero(as_tuple=True)[0]
        solutions[idx, 0] = torch.full((len(idx),), 1.0, device=device, dtype=dtype)

    # Quadratic degenerate case, only fill those (solutions already 0 elsewhere)
    mask_second_order = mask_a_zero & ~mask_b_zero & ~mask_c_zero
    if torch.any(mask_second_order):
        idx = mask_second_order.nonzero(as_tuple=True)[0]
        solutions[idx, 0:2] = solve_quadratic(coeffs[idx, 1:])

    # General cubic, only work on a ≠ 0 cases
    mask_cubic = ~mask_a_zero
    if not torch.any(mask_cubic):
        return solutions  # All degenerate cases, already handled

    cubic_idx = mask_cubic.nonzero(as_tuple=True)[0]
    n_cubic = cubic_idx.shape[0]
    # use only valid cubic batch entries
    a1 = a[cubic_idx]
    b1 = b[cubic_idx]
    c1 = c[cubic_idx]
    d1 = d[cubic_idx]

    inv_a = 1.0 / a1
    b_a = inv_a * b1
    b_a2 = b_a * b_a
    c_a = inv_a * c1
    d_a = inv_a * d1

    Q = (3 * c_a - b_a2) / 9.0
    R = (9.0 * b_a * c_a - 27.0 * d_a - 2.0 * b_a * b_a2) / 54.0
    Q3 = Q * Q * Q
    D = Q3 + R * R
    b_a_3 = (1.0 / 3.0) * b_a

    # Q == 0 and R != 0 => one real root
    mask_Q_zero = (Q == 0) & (R != 0)
    if torch.any(mask_Q_zero):
        idx = mask_Q_zero.nonzero(as_tuple=True)[0]
        x0 = torch.pow(2.0 * R[idx], 1.0 / 3.0) - b_a_3[idx]
        solutions[cubic_idx[idx], 0] = x0

    # Q == 0, R == 0 => triple root
    mask_QR_zero = (Q == 0) & (R == 0)
    if torch.any(mask_QR_zero):
        idx = mask_QR_zero.nonzero(as_tuple=True)[0]
        x0 = -b_a_3[idx]
        v = torch.stack([x0, x0, x0], dim=1)
        solutions[cubic_idx[idx], :] = v

    # D <= 0, Q != 0 => 3 real roots
    mask_D_zero = (D <= 0) & (Q != 0)
    if torch.any(mask_D_zero):
        idx = mask_D_zero.nonzero(as_tuple=True)[0]
        Qidx = Q[idx]
        Ridx = R[idx]
        Q3idx = Q3[idx]
        b_a_3idx = b_a_3[idx]

        sqrt_arg = torch.clamp(-Q3idx, min=0.0)
        acos_arg = torch.clamp(Ridx / torch.sqrt(sqrt_arg + 1e-12), -1.0, 1.0)
        theta = torch.acos(acos_arg)
        sqrt_Q = torch.sqrt(-Qidx)
        x0 = 2 * sqrt_Q * cos(theta / 3.0) - b_a_3idx
        x1 = 2 * sqrt_Q * cos((theta + 2.0 * _PI) / 3.0) - b_a_3idx
        x2 = 2 * sqrt_Q * cos((theta + 4.0 * _PI) / 3.0) - b_a_3idx
        v = torch.stack([x0, x1, x2], dim=1)
        solutions[cubic_idx[idx], :] = v

    # D > 0, Q != 0 => single real root
    mask_D_positive = (D > 0) & (Q != 0)
    if torch.any(mask_D_positive):
        idx = mask_D_positive.nonzero(as_tuple=True)[0]
        Ridx = R[idx]
        Qidx = Q[idx]
        Didx = D[idx]
        b_a_3idx = b_a_3[idx]

        R_abs = torch.abs(Ridx)
        # Only compute where |R| > 1e-16
        mask_R_valid = R_abs > 1e-16
        AD = torch.zeros_like(Ridx)
        BD = torch.zeros_like(Ridx)
        if torch.any(mask_R_valid):
            sqrt_D = torch.sqrt(Didx[mask_R_valid])
            AD_valid = torch.pow(R_abs[mask_R_valid] + sqrt_D, 1.0 / 3.0)
            # sign(R) transfer:
            AD_valid = AD_valid * torch.sign(Ridx[mask_R_valid])
            AD[mask_R_valid] = AD_valid
            BD[mask_R_valid] = -Qidx[mask_R_valid] / AD_valid
        x0 = AD + BD - b_a_3idx
        solutions[cubic_idx[idx], 0] = x0

    return solutions


# def solve_quartic(coeffs: Tensor) -> Tensor:
#    TODO: Quartic equation solver
#     return solutions


# Reference
# https://github.com/danini/graph-cut-ransac/blob/master/src/pygcransac/include/
# estimators/solver_essential_matrix_five_point_nister.h#L108


def multiply_deg_one_poly(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials of the first order [@nister2004efficient].

    Args:
        a: a first order polynomial for variables :math:`(x,y,z,1)`.
        b: a first order polynomial for variables :math:`(x,y,z,1)`.

    Returns:
        degree 2 poly with the order :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.

    """
    return stack(
        [
            a[:, 0] * b[:, 0],
            a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
            a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
            a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0],
            a[:, 1] * b[:, 1],
            a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1],
            a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1],
            a[:, 2] * b[:, 2],
            a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2],
            a[:, 3] * b[:, 3],
        ],
        dim=-1,
    )


# Reference
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L156


def multiply_deg_two_one_poly(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials a and b of degrees two and one [@nister2004efficient].

    Args:
        a: a second degree poly for variables :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.
        b: a first degree poly for variables :math:`(x y z 1)`.

    Returns:
        a third degree poly for variables,
        :math:`(x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2,
        x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1)`.

    """
    return stack(
        [
            a[:, 0] * b[:, 0],
            a[:, 4] * b[:, 1],
            a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
            a[:, 1] * b[:, 1] + a[:, 4] * b[:, 0],
            a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
            a[:, 0] * b[:, 3] + a[:, 3] * b[:, 0],
            a[:, 4] * b[:, 2] + a[:, 5] * b[:, 1],
            a[:, 4] * b[:, 3] + a[:, 6] * b[:, 1],
            a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] + a[:, 5] * b[:, 0],
            a[:, 1] * b[:, 3] + a[:, 3] * b[:, 1] + a[:, 6] * b[:, 0],
            a[:, 2] * b[:, 2] + a[:, 7] * b[:, 0],
            a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2] + a[:, 8] * b[:, 0],
            a[:, 3] * b[:, 3] + a[:, 9] * b[:, 0],
            a[:, 5] * b[:, 2] + a[:, 7] * b[:, 1],
            a[:, 5] * b[:, 3] + a[:, 6] * b[:, 2] + a[:, 8] * b[:, 1],
            a[:, 6] * b[:, 3] + a[:, 9] * b[:, 1],
            a[:, 7] * b[:, 2],
            a[:, 7] * b[:, 3] + a[:, 8] * b[:, 2],
            a[:, 8] * b[:, 3] + a[:, 9] * b[:, 2],
            a[:, 9] * b[:, 3],
        ],
        dim=-1,
    )


# Compute degree 10 poly representing determinant (equation 14 in the paper)
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L368C5-L368C82
def determinant_to_polynomial(A: Tensor) -> Tensor:
    r"""Represent the determinant by the 10th polynomial, used for 5PC solver [@nister2004efficient].

    Args:
        A: Tensor :math:`(*, 3, 13)`.

    Returns:
        a degree 10 poly, representing determinant (Eqn. 14 in the paper).

    """
    cs = zeros(A.shape[0], 11, device=A.device, dtype=A.dtype)
    cs[:, 0] = (
        A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 3]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 12]
    )

    cs[:, 1] = (
        A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 2]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 11]
    )

    cs[:, 2] = (
        A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 7]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 1]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 10]
    )

    cs[:, 3] = (
        A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 7]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 3]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 6]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 10]
    )

    cs[:, 4] = (
        A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 6]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 2]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 7]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 5]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 10]
    )

    cs[:, 5] = (
        A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 5]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 1]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 6]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 6] = (
        A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 5]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 4]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 0]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 7] = (
        A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 8] = (
        A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 9] = (
        A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 0]
    )

    cs[:, 10] = (
        A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 0]
    )

    return cs
