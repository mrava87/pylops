import os
import platform

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import pytest

from pylops.medical import CT2D
from pylops.utils import dottest

par1 = {
    "ny": 51,
    "nx": 30,
    "ntheta": 20,
    "proj_geom_type": "parallel",
    "projector_type": "strip",
    "dtype": "float64",
}  # parallel, strip

par2 = {
    "ny": 51,
    "nx": 30,
    "ntheta": 20,
    "proj_geom_type": "parallel",
    "projector_type": "line",
    "dtype": "float64",
}  # parallel, line

par3 = {
    "ny": 51,
    "nx": 30,
    "ntheta": 20,
    "proj_geom_type": "fanflat",
    "source_origin_dist": 100,
    "origin_detector_dist": 0,
    "projector_type": "strip_fanflat",
    "dtype": "float64",
}  # fanflat, strip

par4 = {
    "ny": 51,
    "nx": 30,
    "ntheta": 20,
    "proj_geom_type": "parallel",
    "projector_type": "cuda",
    "dtype": "float64",
}  # execute using CUDA


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_CT2D(par):
    """Dot-test for CT2D operator"""
    if backend == "cupy" or par["projector_type"] == "cuda":
        pytest.skip("CUDA tests are failing because of severely mismatched adjoint.")

    theta = np.linspace(0.0, np.pi, par["ntheta"], endpoint=False)

    Cop = CT2D(
        (par["ny"], par["nx"]),
        1.0,
        par["ny"],
        theta,
        proj_geom_type=par["proj_geom_type"],
        projector_type=par["projector_type"] if backend == "numpy" else "cuda",
        source_origin_dist=par.get("source_origin_dist", None),
        origin_detector_dist=par.get("origin_detector_dist", None),
        engine="cpu" if backend == "numpy" else "cuda",
    )
    assert dottest(
        Cop,
        par["ny"] * par["ntheta"],
        par["ny"] * par["nx"],
    )
