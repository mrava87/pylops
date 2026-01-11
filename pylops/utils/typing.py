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

IntNDArray = npt.NDArray[np.int_]
NDArray = npt.NDArray
ArrayLike = npt.ArrayLike

InputDimsLike = Union[Sequence[int], IntNDArray]
SamplingLike = Union[Sequence[float], NDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike

if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None

Tavolinearization = Literal["akirich", "fatti", "PS"]
Tderivkind = Literal["forward", "centered", "backward"]
Tengine = Literal["numpy", "cupy", "jax"]
Tinoutengine = Tuple[Tengine, Tengine]
Tparallel_kind = Literal["multiproc", "multithread"]
