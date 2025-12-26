__all__ = [
    "IntNDArray",
    "NDArray",
    "ArrayLike",
    "InputDimsLike",
    "SamplingLike",
    "ShapeLike",
    "DTypeLike",
    "TensorTypeLike",
]

from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops.utils.deps import torch_enabled

if torch_enabled:
    import torch

IntNDArray = npt.NDArray[np.int_]
NumericNDArray = npt.NDArray[np.number]
NDArray = npt.NDArray
ArrayLike = npt.ArrayLike

Int64Vector = np.ndarray[tuple[int], np.dtype[np.int64]]
Float64Vector = np.ndarray[tuple[int], np.dtype[np.float64]]

InputDimsLike = Union[Sequence[int], IntNDArray]
SamplingLike = Union[Sequence[float], NDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike

if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None
