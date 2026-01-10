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


NDArray = npt.NDArray
ArrayLike = npt.ArrayLike
IntNDArray = npt.NDArray[np.integer]
FloatingNDArray = npt.NDArray[np.floating]
InexactNDArray = npt.NDArray[np.inexact]  # float or complex
NumericNDArray = npt.NDArray[np.number]

InputDimsLike = Union[Sequence[int], IntNDArray]
SamplingLike = Union[Sequence[float], FloatingNDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike

if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None
