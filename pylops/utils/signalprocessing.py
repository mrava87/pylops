from __future__ import annotations

__all__ = [
    "convmtx",
    "nonstationary_convmtx",
    "slope_estimate",
    "dip_estimate",
    "pwd_slope_estimate",
]

import warnings
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from pylops.basicoperators import Diagonal, Smoothing2D
from pylops.optimization.leastsquares import preconditioned_inversion
from pylops.utils import deps
from pylops.utils.backend import get_array_module, get_toeplitz
from pylops.utils.typing import NDArray

_jit_message_pwd = deps.numba_import("the plane-wave destruction kernels")

if _jit_message_pwd is None:
    from pylops.signalprocessing._pwd2d_numba import (
        conv_allpass_numba as _conv_allpass_kernel,
    )
else:
    _conv_allpass_kernel = None

_pwd_warning_emitted = False


def _conv_allpass_python(
    din: NDArray, dip: NDArray, order: int, u1: NDArray, u2: NDArray
) -> None:
    """Pure-Python fallback for plane-wave destruction all-pass filtering."""
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
    global _pwd_warning_emitted
    if _conv_allpass_kernel is not None:
        _conv_allpass_kernel(din, dip, order, u1, u2)
    else:
        if not _pwd_warning_emitted and _jit_message_pwd is not None:
            warnings.warn(_jit_message_pwd)
            _pwd_warning_emitted = True
        _conv_allpass_python(din, dip, order, u1, u2)


def convmtx(h: npt.ArrayLike, n: int, offset: int = 0) -> NDArray:
    r"""Convolution matrix

    Makes a dense convolution matrix :math:`\mathbf{C}`
    such that the dot product ``np.dot(C, x)`` is the convolution of
    the filter :math:`h` centered on `offset` and the input signal :math:`x`.

    Equivalent of `MATLAB's convmtx function
    <http://www.mathworks.com/help/signal/ref/convmtx.html>`_ for:
    - ``mode='full'`` when used with ``offset=0``.
    - ``mode='same'`` when used with ``offset=len(h)//2`` (after truncating the rows as ``C[:n]``)

    Parameters
    ----------
    h : :obj:`numpy.ndarray`
        Convolution filter (1D array)
    n : :obj:`int`
        Number of columns of convolution matrix
    offset : :obj:`int`
        Index of the center of the filter

    Returns
    -------
    C : :obj:`numpy.ndarray`
        Convolution matrix of size :math:`\text{len}(h)+n-1 \times n`

    """
    warnings.warn(
        "A new implementation of convmtx is provided in v2.2.0 to match "
        "MATLAB's convmtx method as stated in the docstring. The implementation "
        "of convmtx provided prior to v2.2.0 was instead not consistent "
        "with the documentation. Users are highly encouraged "
        "to modify their codes accordingly.",
        FutureWarning,
    )

    ncp = get_array_module(h)
    nh = len(h)
    col_1 = ncp.r_[h, ncp.zeros(n + nh - 2, dtype=h.dtype)]
    row_1 = ncp.r_[h[0], ncp.zeros(n - 1, dtype=h.dtype)]
    C = get_toeplitz(h)(col_1, row_1)
    # apply offset
    C = C[offset : offset + nh + n - 1]
    return C


def nonstationary_convmtx(
    H: npt.ArrayLike,
    n: int,
    hc: int = 0,
    pad: Tuple[int] = (0, 0),
) -> NDArray:
    r"""Convolution matrix from a bank of filters

    Makes a dense convolution matrix :math:`\mathbf{C}`
    such that the dot product ``np.dot(C, x)`` is the nonstationary
    convolution of the bank of filters :math:`H=[h_1, h_2, h_n]`
    and the input signal :math:`x`.

    Parameters
    ----------
    H : :obj:`numpy.ndarray`
        Convolution filters (2D array of shape
        :math:`[n_\text{filters} \times n_{h}]`
    n : :obj:`int`
        Number of columns of convolution matrix
    hc : :obj:`numpy.ndarray`, optional
        Index of center of first filter
    pad : :obj:`numpy.ndarray`
        Zero-padding to apply to the bank of filters before and after the
        provided values (use it to avoid wrap-around or pass filters with
        enough padding)

    Returns
    -------
    C : :obj:`numpy.ndarray`
        Convolution matrix

    """
    ncp = get_array_module(H)

    H = ncp.pad(H, ((0, 0), pad), mode="constant")
    C = ncp.array([ncp.roll(h, ih) for ih, h in enumerate(H)])
    C = C[:, pad[0] + hc : pad[0] + hc + n].T  # take away edges
    return C


def slope_estimate(
    d: npt.ArrayLike,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
) -> Tuple[NDArray, NDArray]:
    r"""Local slope estimation

    Local slopes are estimated using the *Structure Tensor* algorithm [1]_.
    Slopes are returned as :math:`\tan\theta`, defined
    in a RHS coordinate system with :math:`z`-axis pointing upward.

    .. note:: For stability purposes, it is important to ensure that the orders
        of magnitude of the samplings are similar.

    Parameters
    ----------
    d : :obj:`numpy.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    smooth : :obj:`float` or :obj:`numpy.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.

        .. warning::
            Default changed in version 1.17.0 to 5 from previous value of 20.

    eps : :obj:`float`, optional
        .. versionadded:: 1.17.0

        Regularization term. All slopes where
        :math:`|g_{zx}| < \epsilon \max_{(x, z)} \{|g_{zx}|, |g_{zz}|, |g_{xx}|\}`
        are set to zero. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.

    dips : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Return dips (``True``) instead of slopes (``False``).

    Returns
    -------
    slopes : :obj:`numpy.ndarray`
        Estimated local slopes. The unit is that of
        :math:`\Delta z/\Delta x`.

        .. warning::
            Prior to version 1.17.0, always returned dips.

    anisotropies : :obj:`numpy.ndarray`
        Estimated local anisotropies: :math:`1-\lambda_\text{min}/\lambda_\text{max}`

        .. note::
            Since 1.17.0, changed name from ``linearity`` to ``anisotropies``.
            Definition remains the same.

    Notes
    -----
    For each pixel of the input dataset :math:`\mathbf{d}` the local gradients
    :math:`g_z = \frac{\partial \mathbf{d}}{\partial z}` and
    :math:`g_x = \frac{\partial \mathbf{d}}{\partial x}` are computed
    and used to define the following three quantities:

    .. math::
        \begin{align}
        g_{zz} &= \left(\frac{\partial \mathbf{d}}{\partial z}\right)^2\\
        g_{xx} &= \left(\frac{\partial \mathbf{d}}{\partial x}\right)^2\\
        g_{zx} &= \frac{\partial \mathbf{d}}{\partial z}\cdot\frac{\partial \mathbf{d}}{\partial x}
        \end{align}

    They are then spatially smoothed and at each pixel their smoothed versions are
    arranged in a :math:`2 \times 2` matrix called the *smoothed
    gradient-square tensor*:

    .. math::
        \mathbf{G} =
        \begin{bmatrix}
           g_{zz}  & g_{zx} \\
           g_{zx}  & g_{xx}
        \end{bmatrix}

    Local slopes can be expressed as
    :math:`p = \frac{\lambda_\text{max} - g_{zz}}{g_{zx}}`,
    where :math:`\lambda_\text{max}` is the largest eigenvalue of :math:`\mathbf{G}`.

    Similarly, local dips can be expressed as :math:`\tan(2\theta) = 2g_{zx} / (g_{zz} - g_{xx})`.

    Moreover, we can obtain a measure of local anisotropy, defined as

    .. math::
        a = 1-\lambda_\text{min}/\lambda_\text{max}

    where :math:`\lambda_\text{min}` is the smallest eigenvalue of :math:`\mathbf{G}`.
    A value of :math:`a = 0`  indicates perfect isotropy whereas :math:`a = 1`
    indicates perfect anisotropy.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    slopes = np.zeros_like(d)
    anisos = np.zeros_like(d)

    gz, gx = np.gradient(d, dz, dx)
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    gzz = gaussian_filter(gzz, sigma=smooth)
    gzx = gaussian_filter(gzx, sigma=smooth)
    gxx = gaussian_filter(gxx, sigma=smooth)

    gmax = max(gzz.max(), gxx.max(), np.abs(gzx).max())
    if gmax <= eps:
        return np.zeros_like(d), anisos

    gzz /= gmax
    gzx /= gmax
    gxx /= gmax

    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * np.sqrt((gzz - gxx) ** 2 + 4 * gzx**2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2

    regdata = l1 > eps
    anisos[regdata] = 1 - l2[regdata] / l1[regdata]

    if dips:
        slopes = 0.5 * np.arctan2(2 * gzx, gzz - gxx)
    else:
        regdata = np.abs(gzx) > eps
        slopes[regdata] = (l1 - gzz)[regdata] / gzx[regdata]

    return slopes, anisos


def dip_estimate(
    d: npt.ArrayLike,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: int = 5,
    eps: float = 0.0,
) -> Tuple[NDArray, NDArray]:
    r"""Local dip estimation

    Local dips are estimated using the *Structure Tensor* algorithm [1]_.

    .. note:: For stability purposes, it is important to ensure that the orders
        of magnitude of the samplings are similar.

    Parameters
    ----------
    d : :obj:`numpy.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`
    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`
    smooth : :obj:`float` or :obj:`numpy.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.
    eps : :obj:`float`, optional
        Regularization term. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.

    Returns
    -------
    dips : :obj:`numpy.ndarray`
        Estimated local dips. The unit is radians,
        in the range of :math:`-\frac{\pi}{2}` to :math:`\frac{\pi}{2}`.
    anisotropies : :obj:`numpy.ndarray`
        Estimated local anisotropies: :math:`1-\lambda_\text{min}/\lambda_\text{max}`


    Notes
    -----
    Thin wrapper around ``pylops.utils.signalprocessing.slope_estimate`` with ``dips=True``.
    See the Notes of ``pylops.utils.signalprocessing.slope_estimate`` for details.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    dips, anisos = slope_estimate(d, dz=dz, dx=dx, smooth=smooth, eps=eps, dips=True)
    return dips, anisos


def _triangular_smoothing_from_boxcars(
    nsmooth: Tuple[int, int], dims: Tuple[int, int], dtype: str | np.dtype = "float32"
):
    """Build a triangular smoother as the composition of two boxcar passes."""

    ny, nx = nsmooth
    ly = (ny + 1) // 2
    lx = (nx + 1) // 2

    box = Smoothing2D(nsmooth=(ly, lx), dims=dims, dtype=dtype)
    return box @ box


def pwd_slope_estimate(
    d: ArrayLike,
    niter: int = 5,
    liter: int = 20,
    order: int = 2,
    nsmooth: Tuple[int, int] = (10, 10),
    damp: float = 0.0,
    smoothing: str = "triangle",
) -> NDArray:
    r"""Plane-Wave Destruction (PWD) local slope estimation.

    Slopes :math:`\sigma(z, x)` are estimated following the
    plane-wave-destruction formulation (Claerbout, 1999; Fomel, 2002),
    with optional structure-aligned smoothing preconditioning.

    Parameters
    ----------
    d : :obj:`numpy.ndarray` or :obj:`ArrayLike`
        Input 2D array of shape ``(nz, nx)``.
    niter : :obj:`int`, optional
        Number of outer PWD iterations. Default is ``5``.
    liter : :obj:`int`, optional
        Maximum number of inner least-squares iterations. Default is ``20``.
    order : :obj:`int`, optional
        Accuracy order of the all-pass filters. Use ``1`` (3-tap) or ``2`` (5-tap).
        Default is ``2``.
    nsmooth : :obj:`tuple` of :obj:`int`, optional
        Smoothing lengths ``(ny, nx)`` for the preconditioner. Default ``(10, 10)``.
    damp : :obj:`float`, optional
        Damping factor for the least-squares solve. Default ``0.0``.
    smoothing : :obj:`str`, optional
        Preconditioning choice: ``"triangle"`` (default) applies a triangular
        smoother (two boxcar passes); ``"boxcar"`` applies a single-pass boxcar.

    Returns
    -------
    sigma : :obj:`numpy.ndarray`
        Estimated slope field of shape ``(nz, nx)`` in samples per trace (:math:`\Delta z / \Delta x`).

    Notes
    -----
    ``pwd_slope_estimate`` relies on kernels defined in
    ``pylops.signalprocessing._pwd2d_numba``. When Numba is available the
    implementation is JIT-accelerated; otherwise a pure-Python fallback is used.

    References
    ----------
    - Claerbout, J. F. (1992). *Earth Sounding Analysis: Processing Versus Inversion*.
    - Fomel, S. (2002). "Applications of plane-wave destruction filters."
      *Geophysics*, 67(6), 1946-1960.

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.utils.signalprocessing import pwd_slope_estimate
    >>> nz, nx = 100, 60
    >>> data = np.random.randn(nz, nx).astype("float32")
    >>> sigma = pwd_slope_estimate(data, niter=3, nsmooth=(5, 5))
    >>> sigma.shape
    (100, 60)
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 (B3) or 2 (B5)")

    din = np.asarray(d, dtype=np.float32, order="C")
    if din.ndim != 2:
        raise ValueError("input data must be 2-D")

    nz, nx = din.shape
    sigma = np.zeros((nz, nx), dtype=np.float32)
    delta_sigma = np.zeros_like(sigma)
    u1 = np.zeros_like(sigma)
    u2 = np.zeros_like(sigma)

    smoothing_lower = smoothing.lower()
    if smoothing_lower == "triangle":
        Sop = _triangular_smoothing_from_boxcars(
            nsmooth=nsmooth, dims=(nz, nx), dtype="float32"
        )
    elif smoothing_lower == "boxcar":
        Sop = Smoothing2D(nsmooth=nsmooth, dims=(nz, nx), dtype="float32")
    else:
        raise ValueError("smoothing must be either 'triangle' or 'boxcar'")

    for _ in range(niter):
        _conv_allpass(din, sigma, order, u1, u2)

        Dop = Diagonal(u1.ravel().astype("float32"), dtype="float32")
        delta_sigma[:] = preconditioned_inversion(
            Dop,
            (-u2.ravel()).astype(np.float32, copy=False),
            Sop,
            damp=damp,
            iter_lim=liter,
            show=False,
        )[0].reshape(nz, nx)

        sigma += delta_sigma

    return sigma.astype(np.float32, copy=False)
