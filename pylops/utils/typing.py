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

from typing import Literal, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops.utils.deps import torch_enabled

if torch_enabled:
    import torch


NDArray = npt.NDArray[np.number]
ArrayLike = npt.ArrayLike
IntNDArray = npt.NDArray[np.integer]
FloatingNDArray = npt.NDArray[np.floating]
InexactNDArray = npt.NDArray[np.inexact]  # float or complex

InputDimsLike = Union[Sequence[int], IntNDArray]
SamplingLike = Union[Sequence[float], FloatingNDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike

if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None

Tavolinearization = Literal["akirich", "fatti", "PS"]
Tderivkind = Literal["forward", "centered", "backward"]
Tengine = Literal["numpy", "cupy", "jax"]
Tengine1 = Literal["numpy", "numba", "cuda"]
Tengine2 = Literal["numpy", "numba"]
Tinoutengine = Tuple[Tengine, Tengine]
Tparallel_kind = Literal["multiproc", "multithread"]
Ttaper = Literal["hanning", "cosine", "cosine_square"]
Ttaper = Literal["hanning", "cosine", "cosine_square"]
