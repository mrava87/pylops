__all__ = ["SmoothingND"]

from typing import Union

import numpy as np

from pylops.signalprocessing import ConvolveND
from pylops.utils.typing import DTypeLike, InputDimsLike


class SmoothingND(ConvolveND):
    r"""2D Smoothing.

    Apply smoothing to model (and data) along  the
    ``axes`` of a n-dimensional array.

    Parameters
    ----------
    nsmooth : :obj:`tuple` or :obj:`list`
        Length of smoothing operator in the chosen dimensions
        (must be odd, if even it will be increased by 1).
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes along which model (and data) are smoothed.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    nh : :obj:`tuple`
        Length of the filter
    convolve : :obj:`callable`
        Convolution function
    correlate : :obj:`callable`
        Correlation function
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
    pylops.signalprocessing.ConvolveND : ND convolution

    Notes
    -----
    The ND Smoothing operator is a special type of convolutional operator that
    convolves the input model (or data) with a constant 2d filter of size
    :math:`n_{\text{smooth}, 1} \times n_{\text{smooth}, 2}`:

    Its application to a two dimensional input signal is:

    .. math::
        y[i,j] = 1/(n_{\text{smooth}, 1}\cdot n_{\text{smooth}, 2})
        \sum_{l=-(n_{\text{smooth},1}-1)/2}^{(n_{\text{smooth},1}-1)/2}
        \sum_{m=-(n_{\text{smooth},2}-1)/2}^{(n_{\text{smooth},2}-1)/2} x[l,m]

    Note that since the filter is symmetrical, the *Smoothing2D* operator is
    self-adjoint.

    """

    def __init__(
        self,
        nsmooth: InputDimsLike,
        dims: Union[int, InputDimsLike],
        axes: InputDimsLike = (-2, -1),
        dtype: DTypeLike = "float64",
        name: str = "S",
    ):
        nsmooth = list(nsmooth)
        for i in range(len(nsmooth)):
            if nsmooth[i] % 2 == 0:
                nsmooth[i] += 1
        h = np.ones(nsmooth) / float(np.prod(nsmooth))
        offset = [(nsm - 1) // 2 for nsm in nsmooth]
        super().__init__(
            dims=dims, h=h, offset=offset, axes=axes, dtype=dtype, name=name
        )
