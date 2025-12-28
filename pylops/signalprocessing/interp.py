__all__ = ["Interp"]

import warnings
from typing import Literal, Tuple, Union

import numpy as np

from pylops import LinearOperator, aslinearoperator
from pylops.basicoperators import Diagonal, MatrixMult, Restriction, Transpose
from pylops.signalprocessing.interpspline import CubicSplineInterpolator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module, get_normalize_axis_index
from pylops.utils.typing import (
    DTypeLike,
    Float64Vector,
    InputDimsLike,
    IntNDArray,
    NumericNDArray,
)


def ensure_iava_is_unique(iava: NumericNDArray) -> None:
    _, count = np.unique(iava, return_counts=True)
    if np.any(count > 1):
        raise ValueError("Found repeated values in iava.")

    return


def normalize_iava(
    iava: NumericNDArray,
    iava_max: int,
) -> None:
    # ensure that samples are not beyond the last sample, in that case set to
    # penultimate sample and raise a warning
    outside = iava >= iava_max
    if np.any(outside):
        warnings.warn(
            f"At least one value in iava is beyond the penultimate sample index "
            f"{iava_max}. Out-of-bound-values are forced below penultimate sample"
        )
        # TODO: the following operation is quite dangerous because for example
        # >>> a = 5_000_000
        # >>> a - 1e-10 - a
        # 0.0
        # so for high values of ``iava_max``, this operation is invalidated;
        # it should be iava_max * (1.0 - 1e5 * np.finfo(np.float64).eps) to avoid this
        # and achieve a similar behaviour, but this would be a breaking change
        iava[np.where(outside)] = iava_max - 1e-10

    ensure_iava_is_unique(iava=iava)

    return


def _nearestinterp(
    dims: Union[int, InputDimsLike],
    iava: IntNDArray,
    axis: int = -1,
    dtype: DTypeLike = "float64",
):
    """Nearest neighbour interpolation."""
    iava = np.round(iava).astype(int)
    ensure_iava_is_unique(iava=iava)
    return (Restriction(dims, iava, axis=axis, dtype=dtype), iava)


def _linearinterp(
    dims: InputDimsLike,
    iava: IntNDArray,
    axis: int = -1,
    dtype: DTypeLike = "float64",
):
    """Linear interpolation."""
    ncp = get_array_module(iava)

    if np.issubdtype(iava.dtype, np.integer):
        iava = iava.astype(np.float64)

    lastsample = dims[axis]
    dimsd = list(dims)
    dimsd[axis] = len(iava)
    dimsd = tuple(dimsd)

    # ensure that samples are not beyond the last sample, in that case set to
    # penultimate sample and raise a warning
    normalize_iava(iava=iava, iava_max=lastsample - 1)

    # find indices and weights
    iva_l = ncp.floor(iava).astype(int)
    iva_r = iva_l + 1
    weights = iava - iva_l

    # create operators
    Op = Diagonal(1 - weights, dims=dimsd, axis=axis, dtype=dtype) * Restriction(
        dims, iva_l, axis=axis, dtype=dtype
    ) + Diagonal(weights, dims=dimsd, axis=axis, dtype=dtype) * Restriction(
        dims, iva_r, axis=axis, dtype=dtype
    )
    return Op, iava, dims, dimsd


def _sincinterp(
    dims: InputDimsLike,
    iava: IntNDArray,
    axis: int = 0,
    dtype: DTypeLike = "float64",
):
    """Sinc interpolation."""
    ncp = get_array_module(iava)

    # TODO: is ``iava`` bound to an integer dtype
    ensure_iava_is_unique(iava=iava)

    # create sinc interpolation matrix
    nreg = dims[axis]
    ireg = ncp.arange(nreg)
    sinc = ncp.tile(iava[:, np.newaxis], (1, nreg)) - ncp.tile(ireg, (len(iava), 1))
    sinc = ncp.sinc(sinc)

    # identify additional dimensions and create MatrixMult operator
    otherdims = np.array(dims)
    otherdims = np.roll(otherdims, -axis)
    otherdims = otherdims[1:]
    Op = MatrixMult(sinc, otherdims=otherdims, dtype=dtype)

    # create Transpose operator that brings axis to first dimension
    dimsd = list(dims)
    dimsd[axis] = len(iava)
    if axis > 0:
        axes = np.arange(len(dims), dtype=int)
        axes = np.roll(axes, -axis)
        Top = Transpose(dims, axes=axes, dtype=dtype)
        T1op = Transpose(dimsd, axes=axes, dtype=dtype)
        Op = T1op.H * Op * Top
    return Op, dims, dimsd


def _cubic_spline_interp(
    dims: Tuple,
    iava: NumericNDArray,
    axis: int,
    dtype: DTypeLike,
    name: str,
) -> Tuple[CubicSplineInterpolator, Float64Vector, Tuple, Tuple]:
    """Cubic Spline interpolation"""

    axis = get_normalize_axis_index()(axis, len(dims))

    num_cols = dims[axis]
    if num_cols < 4:
        raise ValueError(
            f"A cubic spline requires at least 4 data points to interpolate, but "
            f"got {dims[axis] = }."
        )

    iava = np.asarray(iava, dtype=np.float64)
    normalize_iava(iava=iava, iava_max=num_cols - 1)

    int64_info = np.iinfo(np.int64)
    if np.any(
        np.logical_or(
            iava < int64_info.min,
            iava > int64_info.max,
        )
    ):
        raise OverflowError("iava contains indices that make numpy.int64 overflow.")

    dtype = np.dtype(dtype)
    if dtype.type not in {np.float64, np.complex128}:
        raise TypeError(
            f"Expected dtype fo cubic spline interpolator to be either float64 or "
            f"complex128 to achieve the required accuracy, but got {dtype}."
        )

    # --- Setup ---

    dimsd = list(dims)
    dimsd[axis] = len(iava)
    dimsd = tuple(dimsd)

    Op = CubicSplineInterpolator(
        dims=dims,
        dimsd=dimsd,
        iava=iava,
        axis=axis,  # type: ignore
        dtype=dtype.type,
        name=name,
    )

    return Op, iava, dims, dimsd


def Interp(
    dims: Union[int, InputDimsLike],
    iava: IntNDArray,
    axis: int = -1,
    kind: Literal["linear", "nearest", "sinc", "cubic_spline"] = "linear",
    dtype: DTypeLike = "float64",
    name: str = "I",
) -> Tuple[LinearOperator, IntNDArray]:
    r"""Interpolation operator.

    Apply interpolation along ``axis`` from regularly sampled input
    vector into fractionary positions ``iava`` using one of the
    following algorithms:

    - *Nearest neighbour* interpolation
      is a thin wrapper around :obj:`pylops.Restriction` at ``np.round(iava)``
      locations.

    - *Linear interpolation* extracts values from input vector
      at locations ``np.floor(iava)`` and ``np.floor(iava)+1`` and linearly
      combines them in forward mode, places weighted versions of the
      interpolated values at locations ``np.floor(iava)`` and
      ``np.floor(iava)+1`` in an otherwise zero vector in adjoint mode.

    - *Sinc interpolation* performs sinc interpolation at locations ``iava``.
      Note that this is the most accurate method but it has higher computational
      cost as it involves multiplying the input data by a matrix of size
      :math:`N \times M`.

    - *Cubic Spline interpolation* relies on a cubic spline, i.e., a 2-times
      continuously differentiable piecewise third order polynomial with equally spaced
      knots. It is interpolated at the locations ``iava`` by evaluating the respective
      polynomial fitted between ``np.floor(iava)`` and ``np.floor(iava) + 1``.
      It offers an excellent tradeoff between accuracy and computational complexity
      and its results oscillate less than those obtained from sinc interpolation.

    .. note:: The vector ``iava`` should contain unique values. If the same
      index is repeated twice an error will be raised. This also applies when
      values beyond the last element of the input array for
      *linear interpolation* and *cubic spline interpolation* as those values are forced
      to be just before this element.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    iava : :obj:`list` or :obj:`numpy.ndarray`
         Floating indices of locations of available samples for interpolation.
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which interpolation is applied.
    kind : :obj:`str`, optional
        Kind of interpolation.
        Currently, ``"nearest"``, ``"linear"``, ``"sinc"``, and ``"cubic_spline"`` are
        available.

        .. versionadded:: 2.0.0

        The ``"cubic_spline"``-interpolation was added.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    op : :obj:`pylops.LinearOperator`
        Linear intepolation operator
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Corrected indices of locations of available samples
        (samples at ``M-1`` or beyond are forced to be at ``M-1-eps``)

    Raises
    ------
    ValueError
        If the vector ``iava`` contains repeated values.
    NotImplementedError
        If ``kind`` is not ``nearest``, ``linear`` or ``sinc``

    See Also
    --------
    pylops.Restriction : Restriction operator

    Notes
    -----
    *Linear interpolation* of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = (1-w_i) x_{l^{l}_i} + w_i x_{l^{r}_i}
        \quad \forall i=1,2,\ldots,N

    where :math:`\mathbf{l^l}=[\lfloor l_1 \rfloor, \lfloor l_2 \rfloor,\ldots,
    \lfloor l_N \rfloor]` and :math:`\mathbf{l^r}=[\lfloor l_1 \rfloor +1,
    \lfloor l_2 \rfloor +1,\ldots,
    \lfloor l_N \rfloor +1]` are vectors containing the indeces
    of the original array at which samples are taken, and
    :math:`\mathbf{w}=[l_1 - \lfloor l_1 \rfloor, l_2 - \lfloor l_2 \rfloor,
    ..., l_N - \lfloor l_N \rfloor]` are the linear interpolation weights.
    This operator can be implemented by simply summing two
    :class:`pylops.Restriction` operators which are weighted
    using :class:`pylops.basicoperators.Diagonal` operators.

    *Sinc interpolation* of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::
        \DeclareMathOperator{\sinc}{sinc}
        y_i = \sum_{j=0}^{M} x_j \sinc(i-j) \quad \forall i=1,2,\ldots,N

    This operator can be implemented using the :class:`pylops.MatrixMult`
    operator with a matrix containing the values of the sinc function at all
    :math:`i,j` possible combinations.

    """
    dims = _value_or_sized_to_tuple(dims)

    if kind == "nearest":
        interpop, iava = _nearestinterp(dims, iava, axis=axis, dtype=dtype)
    elif kind == "linear":
        interpop, iava, dims, dimsd = _linearinterp(dims, iava, axis=axis, dtype=dtype)
    elif kind == "sinc":
        interpop, dims, dimsd = _sincinterp(dims, iava, axis=axis, dtype=dtype)
    elif kind == "cubic_spline":
        (
            interpop,
            iava,
            dims,
            dimsd,
        ) = _cubic_spline_interp(
            dims=dims,
            iava=iava,
            axis=axis,  # type: ignore
            dtype=dtype,
            name=name,
        )

    else:
        raise NotImplementedError(f"{kind} interpolation could not be found.")
    # add dims and dimsd to composite operators (not needed for neareast as
    # interpop is a Restriction operator already
    if kind not in {"nearest"}:
        interpop = aslinearoperator(interpop)
        interpop.dims = dims
        interpop.dimsd = dimsd
        interpop.name = name
    return interpop, iava
