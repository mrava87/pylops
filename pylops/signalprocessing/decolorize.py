__all__ = ["Decolorize"]

from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Identity, HStack
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Decolorize(LinearOperator):
    r"""Decolorize operator.

    Converts RGB-channel signals to grayscale. The different channels are 
    multiplied by factors determined by the chosen spectral response function 
    (SRF), then summed to produce a grayscale image.

    Parameters
    ----------
    
    Attributes
    ----------

    Raises
    ------
    NotImplementedError
        If ``srf`` is neither ``rec601``, ``ave`` nor ``random``.

    Notes
    -----
    XX

    """

    def __init__(
        self,
        dims: InputDimsLike,
        srf: str = "rec601",
        dtype: DTypeLike = "float64",
        name: str = "D",
    ) -> None:
        dims = _value_or_sized_to_tuple(dims)
        if srf == "rec601":
            coeffs = np.array([0.2989, 0.5870, 0.1140], dtype=dtype)
        elif srf == "ave":
            coeffs = np.full(3, 1. / 3., dtype=dtype)
        elif srf == "random":
            coeffs = np.random.rand(3).astype(dtype)
            coeffs /= coeffs.sum()
        else:
            raise NotImplementedError("srf must be rec601, ave, or random")
        Op = self._calc_op(
            dims=dims,
            coeffs=coeffs,
            dtype=dtype,
        )
        super().__init__(Op=Op, name=name)

    def _matvec(self, x: NDArray) -> NDArray:
        return super()._matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return super()._rmatvec(x)

    @staticmethod
    def _calc_op(
        dims: InputDimsLike,
        coeffs:  NDArray,
        dtype: DTypeLike,
    ):
        iop = Identity(dims, dtype=dtype)
        op = HStack([coeffs[0] * iop, coeffs[1] * iop, coeffs[2] * iop])
        op.dims = (3, *list(dims))
        print(op.dims, op.dimsd)
        return op
