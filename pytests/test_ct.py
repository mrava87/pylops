import numpy as np
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
    "projector_type": "strip",
    "dtype": "float64",
}  # fanflat, strip


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_CT2D(par):
    """Dot-test for CT2D operator"""
    theta = np.linspace(0.0, np.pi, par["ntheta"], endpoint=False)

    Cop = CT2D(
        (par["ny"], par["nx"]),
        1.0,
        par["ny"],
        theta,
        proj_geom_type=par["proj_geom_type"],
        projector_type=par["projector_type"],
    )
    assert dottest(
        Cop,
        par["ny"] * par["ntheta"],
        par["ny"] * par["nx"],
    )
