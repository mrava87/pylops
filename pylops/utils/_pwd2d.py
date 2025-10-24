import warnings
from typing import Tuple

import numpy as np

from pylops.basicoperators import Smoothing2D
from pylops.utils import deps
from pylops.utils.typing import NDArray

jit_message = deps.numba_import("the plane-wave destruction numba kernels")

if jit_message is None:
    from pylops.utils._pwd2d_numba import _conv_allpass_numba
else:
    _conv_allpass_numba = None


def _conv_allpass_python(
    din: NDArray, dip: NDArray, order: int, u1: NDArray, u2: NDArray
) -> None:
    """Pure-Python fallback for PWD all-pass filtering used in
    :func:`pylops.utils.signalprocessing.pwd_slope_estimate`."""
    n1, n2 = din.shape
    nw = 1 if order == 1 else 2

    for j in range(n1):
        for i in range(n2):
            u1[j, i] = 0.0
            u2[j, i] = 0.0

    def b3(sig: float):
        b0 = (1.0 - sig) * (2.0 - sig) / 12.0
        b1 = (2.0 + sig) * (2.0 - sig) / 6.0
        b2 = (1.0 + sig) * (2.0 + sig) / 12.0
        return b0, b1, b2

    def b3d(sig: float):
        b0 = -(2.0 - sig) / 12.0 - (1.0 - sig) / 12.0
        b1 = (2.0 - sig) / 6.0 - (2.0 + sig) / 6.0
        b2 = (2.0 + sig) / 12.0 + (1.0 + sig) / 12.0
        return b0, b1, b2

    def b5(sig: float):
        s = sig
        b0 = (1 - s) * (2 - s) * (3 - s) * (4 - s) / 1680.0
        b1 = (4 - s) * (2 - s) * (3 - s) * (4 + s) / 420.0
        b2 = (4 - s) * (3 - s) * (3 + s) * (4 + s) / 280.0
        b3 = (4 - s) * (2 + s) * (3 + s) * (4 + s) / 420.0
        b4 = (1 + s) * (2 + s) * (3 + s) * (4 + s) / 1680.0
        return b0, b1, b2, b3, b4

    def b5d(sig: float):
        s = sig
        b0 = (
            -(
                (2 - s) * (3 - s) * (4 - s)
                + (1 - s) * (3 - s) * (4 - s)
                + (1 - s) * (2 - s) * (4 - s)
                + (1 - s) * (2 - s) * (3 - s)
            )
            / 1680.0
        )
        b1 = (
            -(
                (2 - s) * (3 - s) * (4 + s)
                + (4 - s) * (3 - s) * (4 + s)
                + (4 - s) * (2 - s) * (4 + s)
            )
            / 420.0
            + (4 - s) * (2 - s) * (3 - s) / 420.0
        )
        b2 = (
            -((3 - s) * (3 + s) * (4 + s) + (4 - s) * (3 + s) * (4 + s)) / 280.0
            + (4 - s) * (3 - s) * (4 + s) / 280.0
            + (4 - s) * (3 - s) * (3 + s) / 280.0
        )
        b3 = (
            -((2 + s) * (3 + s) * (4 + s)) / 420.0
            + (4 - s) * (3 + s) * (4 + s) / 420.0
            + (4 - s) * (2 + s) * (4 + s) / 420.0
            + (4 - s) * (2 + s) * (3 + s) / 420.0
        )
        b4 = (
            (2 + s) * (3 + s) * (4 + s)
            + (1 + s) * (3 + s) * (4 + s)
            + (1 + s) * (2 + s) * (4 + s)
            + (1 + s) * (2 + s) * (3 + s)
        ) / 1680.0
        return b0, b1, b2, b3, b4

    for i1 in range(nw, n1 - nw):
        for i2 in range(0, n2 - 1):
            s = dip[i1, i2]
            if order == 1:
                b0d, b1d, b2d = b3d(s)
                b0, b1, b2 = b3(s)
                v = din[i1 - 1, i2 + 1] - din[i1 + 1, i2]
                u1[i1, i2] += v * b0d
                u2[i1, i2] += v * b0
                v = din[i1 + 0, i2 + 1] - din[i1 + 0, i2]
                u1[i1, i2] += v * b1d
                u2[i1, i2] += v * b1
                v = din[i1 + 1, i2 + 1] - din[i1 - 1, i2]
                u1[i1, i2] += v * b2d
                u2[i1, i2] += v * b2
            else:
                c0d, c1d, c2d, c3d, c4d = b5d(s)
                c0, c1, c2, c3, c4 = b5(s)
                v = din[i1 - 2, i2 + 1] - din[i1 + 2, i2]
                u1[i1, i2] += v * c0d
                u2[i1, i2] += v * c0
                v = din[i1 - 1, i2 + 1] - din[i1 + 1, i2]
                u1[i1, i2] += v * c1d
                u2[i1, i2] += v * c1
                v = din[i1 + 0, i2 + 1] - din[i1 + 0, i2]
                u1[i1, i2] += v * c2d
                u2[i1, i2] += v * c2
                v = din[i1 + 1, i2 + 1] - din[i1 - 1, i2]
                u1[i1, i2] += v * c3d
                u2[i1, i2] += v * c3
                v = din[i1 + 2, i2 + 1] - din[i1 - 2, i2]
                u1[i1, i2] += v * c4d
                u2[i1, i2] += v * c4


def _conv_allpass(
    din: NDArray, dip: NDArray, order: int, u1: NDArray, u2: NDArray
) -> None:
    """Dispatch to numba kernel when available, otherwise use Python fallback."""
    if _conv_allpass_numba is not None:
        _conv_allpass_numba(din, dip, order, u1, u2)
    else:
        warnings.warn(jit_message)
        _conv_allpass_python(din, dip, order, u1, u2)


def _triangular_smoothing_from_boxcars(
    nsmooth: Tuple[int, int],
    dims: Tuple[int, int],
    dtype: str | np.dtype = "float64",
):
    """Build a triangular smoother as the composition of two boxcar passes."""

    ny, nx = nsmooth
    ly = (ny + 1) // 2
    lx = (nx + 1) // 2

    box = Smoothing2D(nsmooth=(ly, lx), dims=dims, dtype=dtype)
    return box @ box
