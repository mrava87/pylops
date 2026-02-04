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

# numpy generic types
NDArray = npt.NDArray[np.number]
ArrayLike = npt.ArrayLike
IntNDArray = npt.NDArray[np.integer]
FloatingNDArray = npt.NDArray[np.floating]
InexactNDArray = npt.NDArray[np.inexact]  # float or complex

InputDimsLike = Union[Sequence[int], IntNDArray]
SamplingLike = Union[Sequence[float], FloatingNDArray]
ShapeLike = Tuple[int, ...]
DTypeLike = npt.DTypeLike

# torch generic types
if torch_enabled:
    TensorTypeLike = torch.Tensor
else:
    TensorTypeLike = None

# pylops specific types
Tengine = Literal["numpy", "cupy", "jax"]
Tengine1 = Literal["numpy", "numba", "cuda"]
Tengine2 = Literal["numpy", "numba"]
Tctengine = Literal["cpu", "cuda"]
Tfftengine = Literal["numpy", "scipy", "fftw"]
Tfftengine2 = Literal["numpy", "scipy"]
Tfftengine3 = Literal["numpy", "scipy", "mkl_fft"]
Tinoutengine = Tuple[Tengine, Tengine]
Tmriengine = Literal["numpy", "jax"]

Tavolinearization = Literal["akirich", "fatti", "PS"]
Tderivkind = Literal["forward", "centered", "backward"]
Tparallel_kind = Literal["multiproc", "multithread"]
Ttaper = Literal["hanning", "cosine", "cosine_square"]
Tctprojgeom = Literal["parallel", "fanflat"]
Tctprojectortype = Literal["strip", "line", "linear", "cuda"]
Tmrimask = Literal["vertical-reg", "vertical-uni", "radial-reg", "radial-uni"]

Tbackend = Literal["numpy", "cupy"]
Tmemunit = Literal["B", "KB", "MB", "GB"]
Tsolverengine = Literal["scipy", "pylops"]
Tirlskind = Literal["data", "model", "datamodel"]
Tthreshkind = Literal[
    "hard",
    "soft",
    "half",
    "hard-percentile",
    "soft-percentile",
    "half-percentile",
]

Tsampler = Literal["gaussian", "rayleigh", "rademacher", "unitvector"]
Tsampler2 = Literal["gaussian", "rayleigh", "rademacher"]
Tpwdsmoothing = Literal["triangle", "boxcar"]
