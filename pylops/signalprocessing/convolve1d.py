__all__ = ["Convolve1D"]

from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_array, _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_convolve,
    get_fftconvolve,
    get_oaconvolve,
    to_cupy_conditional,
)
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


def _choose_convfunc(
    x: npt.ArrayLike, method: Union[None, str], dims
) -> Tuple[Callable, str]:
    """Choose convolution function

    Choose and return the function handle to be used for convolution
    """
    if len(dims) == 1:
        if method is None:
            method = "direct"
        if method not in ("direct", "fft"):
            raise NotImplementedError("method must be direct or fft")
        convfunc = get_convolve(x)
    else:
        if method is None:
            method = "fft"
        if method == "fft":
            convfunc = get_fftconvolve(x)
        elif method == "overlapadd":
            convfunc = get_oaconvolve(x)
        else:
            raise NotImplementedError("method must be fft or overlapadd")
    return convfunc, method


class _Convolve1Dshort(LinearOperator):
    r"""1D convolution operator with compact filter (shorter than input signal)"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        h: NDArray,
        offset: int = 0,
        axis: int = -1,
        method: str = None,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.axis = axis
        if offset > len(h) - 1:
            raise ValueError("offset must be smaller than len(h) - 1")
        self.nh = len(h)
        self.h = h
        self.hstar = np.flip(self.h, axis=-1)
        self.offset = 2 * (self.nh // 2 - int(offset))
        if self.nh % 2 == 0:
            self.offset -= 1
        if self.offset != 0:
            self.h = np.pad(
                self.h,
                (
                    self.offset if self.offset > 0 else 0,
                    -self.offset if self.offset < 0 else 0,
                ),
                mode="constant",
            )
        self.hstar = np.flip(self.h, axis=-1)

        # add dimensions to filter to match dimensions of model and data
        hdims = np.ones(len(self.dims), dtype=int)
        hdims[self.axis] = len(self.h)
        self.h = self.h.reshape(hdims)
        self.hstar = self.hstar.reshape(hdims)

        # choose method and function handle
        self.convfunc, self.method = _choose_convfunc(h, method, self.dims)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convfunc, self.method = _choose_convfunc(
                self.h, self.method, self.dims
            )
        return self.convfunc(x, self.h, mode="same")

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        if type(self.hstar) != type(x):
            self.hstar = to_cupy_conditional(x, self.hstar)
            self.convfunc, self.method = _choose_convfunc(
                self.hstar, self.method, self.dims
            )
        return self.convfunc(x, self.hstar, mode="same")


class _Convolve1Dlong(LinearOperator):
    r"""1D convolution operator with extended filter (larger than input signal)"""

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        h: NDArray,
        offset: int = 0,
        axis: int = -1,
        method: str = None,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        dimsd = _value_or_sized_to_array(dims)
        dims = _value_or_sized_to_tuple(dims)
        dimsd[axis] = len(h)
        dimsd = _value_or_sized_to_tuple(dimsd)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        # create filter
        self.dimeven = True if dims[axis] == 0 else False
        self.heven = True if len(h) % 2 == 0 else False
        if self.heven:
            # force filter to have odd size
            h = np.pad(h, (0, 1))

        self.axis = axis
        if offset > len(h) - 1:
            raise ValueError("offset must be smaller than len(h) - 1")
        self.nh = len(h)
        self.h = h
        self.offset = 2 * (self.dims[self.axis] // 2 - int(offset))
        if self.offset != 0:
            self.h = np.pad(
                self.h,
                (
                    self.offset if self.offset > 0 else 0,
                    -self.offset if self.offset < 0 else 0,
                ),
                mode="constant",
            )
        self.hstar = np.flip(self.h, axis=-1)
        self.pad = np.zeros((len(dims), 2), dtype=int)
        self.pad[self.axis, 1] = 1

        # add dimensions to filter to match dimensions of model and data
        hdims = np.ones(len(self.dims), dtype=int)
        hdims[self.axis] = len(self.h)
        self.h = self.h.reshape(hdims)
        self.hstar = self.hstar.reshape(hdims)

        # choose how to cut signals
        self.start, self.end = self.offset, self.dimsd[self.axis] + self.offset
        self.startd, self.endd = self.offset, self.dims[self.axis] + self.offset
        if self.dimeven:
            self.start += 1
            self.end += 1
        if self.heven:
            self.startd += 1
            self.endd += 1

        # choose method and function handle
        self.convfunc, self.method = _choose_convfunc(h, method, self.dims)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convfunc, self.method = _choose_convfunc(
                self.h, self.method, self.dims
            )
        x = np.pad(x, self.pad)
        y = self.convfunc(self.h, x, mode="same")
        y = ncp.take(y, range(self.start, self.end), axis=self.axis)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if type(self.hstar) != type(x):
            self.hstar = to_cupy_conditional(x, self.hstar)
            self.convfunc, self.method = _choose_convfunc(
                self.hstar, self.method, self.dims
            )
        y = self.convfunc(self.hstar, x, mode="same")
        print(y)
        y = ncp.take(y, range(self.startd, self.endd), axis=self.axis)
        return y


def Convolve1D(
    dims: Union[int, InputDimsLike],
    h: NDArray,
    offset: int = 0,
    axis: int = -1,
    method: str = None,
    dtype: DTypeLike = "float64",
    name: str = "C",
) -> LinearOperator:
    """1D convolution operator.

    Apply one-dimensional convolution with a compact filter to model (and data)
    along an ``axis`` of a multi-dimensional array.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    h : :obj:`numpy.ndarray`
        1d compact filter to be convolved to input signal
    offset : :obj:`int`
        Index of the center of the compact filter
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which convolution is applied
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct``, ``fft``,
        or ``overlapadd``). Note that only ``direct`` and ``fft`` are allowed
        when ``dims=None``, whilst ``fft`` and ``overlapadd`` are allowed
        when ``dims`` is provided.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``offset`` is bigger than ``len(h) - 1``
    NotImplementedError
        If ``method`` provided is not allowed

    Notes
    -----
    The Convolve1D operator applies convolution between the input signal
    :math:`x(t)` and a compact filter kernel :math:`h(t)` in forward model:

    .. math::
        y(t) = \int\limits_{-\infty}^{\infty} h(t-\tau) x(\tau) \,\mathrm{d}\tau

    This operation can be discretized as follows

    .. math::
        y[n] = \sum_{m=-\infty}^{\infty} h[n-m] x[m]

    as well as performed in the frequency domain.

    .. math::
        Y(f) = \mathscr{F} (h(t)) * \mathscr{F} (x(t))

    Convolve1D operator uses :py:func:`scipy.signal.convolve` that
    automatically chooses the best domain for the operation to be carried out
    for one dimensional inputs. The fft implementation
    :py:func:`scipy.signal.fftconvolve` is however enforced for signals in
    2 or more dimensions as this routine efficently operates on
    multi-dimensional arrays.

    As the adjoint of convolution is correlation, Convolve1D operator applies
    correlation in the adjoint mode.

    In time domain:

    .. math::
        x(t) = \int\limits_{-\infty}^{\infty} h(t+\tau) x(\tau) \,\mathrm{d}\tau

    or in frequency domain:

    .. math::
        y(t) = \mathscr{F}^{-1} (H(f)^* * X(f))

    """
    if len(h) <= _value_or_sized_to_tuple(dims)[axis]:
        convop = _Convolve1Dshort
    else:
        convop = _Convolve1Dlong
    c = convop(
        dims=dims, h=h, offset=offset, axis=axis, method=method, dtype=dtype, name=name
    )
    return c
