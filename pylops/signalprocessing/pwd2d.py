__all__ = ["PWSprayer2D", "PWSmoother2D"]

from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, NDArray

try:
    from ._spraying2d_numba import spray_adjoint_numba, spray_forward_numba
except ImportError:
    from _spraying2d_numba import spray_adjoint_numba, spray_forward_numba


class PWSprayer2D(LinearOperator):
    r"""2D Plane-Wave Sprayer.

    Spray (or paint) each input value along local structural
    slopes in :math:`\pm x` direction with exponential decay
    in forward mode, and gather contributions back along the same
    slope trajectories in adjoint mode. Together, this pair of
    operators defines a linear operator that propagates information
    preferentially along structural dips (implemented in
    :class:`pylops.signalprocessing.PWSmoother2D`).

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int`
        Number of samples for each dimension - ``(nz, nx)``.
    sigma : :obj:`numpy.ndarray`
        Local slope field of shape ``(nz, nx)`` defined in samples per trace
        (:math:`dz/dx`).
    radius : :obj:`int`, optional
        Maximum number of steps along each :math:`\pm x` direction to spray
        or gather. Controls the spatial extent of spreading. Default is ``8``.
    alpha : :obj:`float`, optional
        Geometric decay factor per step (:math:`0 < \alpha \leq 1`).
        Higher values propagate energy farther. Default is ``0.9``.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

    See Also
    --------
    PWSmoother2D : Structure-aligned smoother
    pwd_slope_estimate : Local slope estimation using plane-wave destruction

    Notes
    -----
    - The forward operator distributes each sample along rays following
      local slope :math:`\sigma(z,x)`, using bilinear interpolation in depth.
    - The adjoint operator gathers contributions back from the same rays,
      making this a true linear operator pair.
    - Effective smoothing along dip grows with larger ``radius`` and
      higher ``alpha``.

    """

    def __init__(
        self,
        dims: Tuple[int, int],
        sigma: NDArray,
        radius: int = 8,
        alpha: float = 0.9,
        dtype: DTypeLike = "float64",
        name: str = "P",
    ):
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two elements (nz, nx)")
        self._radius = int(radius)
        self._alpha = float(alpha)
        self._sigma = np.ascontiguousarray(sigma)
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
    r"""2D Structure-aligned smoother.

    This operator builds a symmetric, positive semi-definite (PSD) smoother
    aligned over local structural dips (``sigma``). It is defined as:

    .. math::

        S = P^H P

    where :math:`P` is a :class:`pylops.signalprocessing.PWSprayer2D` operator that
    propagates values along local slopes. The composition ``P.H @ P`` produces a
    correlation-like operator that smooths preferentially along the defined
    slopes.

    The resulting operator can be used as a regularizer or preconditioner
    in inverse problems to enforce structural smoothness.

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int`
        Number of samples for each dimension - ``(nz, nx)``.
    sigma : :obj:`numpy.ndarray`
        Local slope field of shape ``(nz, nx)`` defined in samples per trace
        (:math:`dz/dx`).
    radius : :obj:`int`, optional
        Maximum number of steps along each :math:`\pm x` direction to spray
        or gather. Controls the spatial extent of spreading. Default is ``8``.
    alpha : :obj:`float`, optional
        Geometric decay factor per step (:math:`0 < \alpha \leq 1`).
        Higher values propagate energy farther. Default is ``0.9``.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

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

    """

    def __init__(
        self,
        dims: Tuple[int, int],
        sigma: NDArray,
        radius: int = 8,
        alpha: float = 0.9,
        dtype: DTypeLike = "float64",
        name: str = "P",
    ):
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two elements (nz, nx)")
        self._sprayer = PWSprayer2D(
            dims=dims, sigma=sigma, radius=radius, alpha=alpha, dtype=dtype
        )
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._sprayer @ x
        y = self._sprayer.H @ y
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        return self._matvec(x)
