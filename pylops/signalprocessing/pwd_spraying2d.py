"""Plane-wave spraying operators for structure-aligned smoothing."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils.decorators import reshaped
from pylops.utils.typing import NDArray

try:  
    from ._spraying2d_numba import spray_forward_numba, spray_adjoint_numba
except ImportError:  
    from _spraying2d_numba import spray_forward_numba, spray_adjoint_numba

__all__ = ["PWSprayer2D", "PWSmoother2D"]

class PWSprayer2D(LinearOperator):
    
    r"""
    Plane-Wave Sprayer in 2D.

    Forward mode sprays (paints) each input value along local structural
    slopes in the :math:`\pm x` directions with exponential decay.
    The adjoint mode gathers contributions back along the same slope
    trajectories. Together, these define a linear operator that propagates
    information preferentially along structural dips.

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int`
        Dimensions of the 2D model (``nz, nx``).
    sigma : :obj:`numpy.ndarray`
        Local slope field of shape ``(nz, nx)``, in samples per trace
        (:math:`dz/dx`).
    radius : :obj:`int`, optional
        Maximum number of steps along each :math:`\pm x` direction to spray
        or gather. Controls the spatial extent of spreading. Default is ``8``.
    alpha : :obj:`float`, optional
        Geometric decay factor per step (:math:`0 < \alpha \leq 1`).
        Higher values propagate energy farther. Default is ``0.9``.
    dtype : :obj:`str` or :obj:`numpy.dtype`, optional
        Data type of the operator. Default is ``'float32'``.

    Notes
    -----
    - The forward operator distributes each sample along rays following
      local slope :math:`\sigma(z,x)`, using bilinear interpolation in depth.
    - The adjoint operator gathers contributions back from the same rays,
      making this a true linear operator pair.
    - Effective smoothing along dip grows with larger ``radius`` and
      higher ``alpha``.

    See Also
    --------
    PWSmoother2D : Structure-aligned smoother (``Sprayer.T @ Sprayer``)
    pwd_slope_estimate : Local slope estimation using plane-wave destruction

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.utils import dottest
    >>> nz, nx = 40, 30
    >>> sigma = np.zeros((nz, nx), dtype='float32')  # flat slope
    >>> P = PWSprayer2D(dims=(nz, nx), sigma=sigma, radius=4, alpha=0.9)
    >>> x = np.zeros(nz*nx, dtype='float32')
    >>> x[nz//2 * nx + nx//2] = 1.0   # impulse in the center
    >>> y = P @ x                     # spray along horizontal direction
    >>> dottest(P, nz*nx, nz*nx, complexflag=False)
    True
    """

    def __init__(
        self,
        dims: Tuple[int, int],
        sigma: NDArray,
        radius: int = 8,
        alpha: float = 0.9,
        dtype: str | np.dtype = "float32",
        name: str = "PWSp",
    ):
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two elements (nz, nx)")
        self._radius = int(radius)
        self._alpha = float(alpha)
        self._sigma = np.ascontiguousarray(sigma.astype(np.float32))
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = np.zeros_like(x)
        spray_forward_numba(x, self._sigma, self._radius, self._alpha, y)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = np.zeros_like(x)
        spray_adjoint_numba(x, self._sigma, self._radius, self._alpha, y)
        return y



class PWSmoother2D(LinearOperator):
    
    r"""
    Structure-aligned 2D smoother based on plane-wave spraying.

    This operator builds a symmetric, positive semi-definite (PSD) smoother
    aligned with local structural dips. It is defined as

    .. math::

        S = P^\top P

    where :math:`P` is a :class:`PWSprayer2D` operator that propagates
    values along local slopes. The composition ``P.T @ P`` produces a
    correlation-like operator that smooths preferentially along dip
    directions.

    The resulting operator can be used as a regularizer or preconditioner
    in inverse problems to enforce structural smoothness.

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int`
        Dimensions of the 2D model (``nz, nx``).
    sigma : :obj:`numpy.ndarray`
        Local slope field of shape ``(nz, nx)``, in samples per trace
        (:math:`dz/dx`).
    radius : :obj:`int`, optional
        Maximum number of steps (in samples) to spray along ``±x``.
        Default is ``8``.
    alpha : :obj:`float`, optional
        Geometric decay factor per step (:math:`0<\alpha\leq 1`).
        Controls effective smoothing length. Default is ``0.9``.
    dtype : :obj:`str` or :obj:`numpy.dtype`, optional
        Data type of the operator. Default is ``'float32'``.

    Notes
    -----
    - The smoother is symmetric by construction.
    - Effective correlation length along dip increases with larger
      ``radius`` and higher ``alpha``.
    - Across-dip coupling is minimal (only through interpolation and dip
      variability).

    See Also
    --------
    PWSprayer2D : Forward sprayer/gather operator
    pwd_slope_estimate : Local slope estimation using plane-wave destruction

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.utils import dottest
    >>> nz, nx = 50, 30
    >>> sigma = np.zeros((nz, nx), dtype='float32')  # flat structure
    >>> Sop = PWSmoother2D(dims=(nz, nx), sigma=sigma, radius=4, alpha=0.9)
    >>> x = np.random.randn(nz*nx).astype('float32')
    >>> y = Sop @ x
    >>> dottest(Sop, nz*nx, nz*nx, complexflag=False)
    True
    """
    def __init__(
        self,
        dims: Tuple[int, int],
        sigma: NDArray,
        radius: int = 8,
        alpha: float = 0.9,
        dtype: str | np.dtype = "float32",
        name: str = "PWSm",
    ):
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two elements (nz, nx)")
        self._sprayer = PWSprayer2D(
            dims=dims, sigma=sigma, radius=radius, alpha=alpha, dtype=dtype
        )
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        # y = Spray^T (Spray x)
        y = self._sprayer @ x
        y = self._sprayer.H @ y
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        # symmetric
        return self._matvec(x)
