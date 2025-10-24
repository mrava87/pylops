import os

import numpy as np
import pytest

from pylops.signalprocessing import PWSmoother2D, PWSprayer2D
from pylops.utils import dottest

par1 = {
    "nx": 16,
    "nt": 30,
    "dtype": "float64",
}  # even
par2 = {
    "nx": 17,
    "nt": 31,
    "dtype": "float64",
}  # odd

np.random.seed(10)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PWSprayer2D(par):
    """Dot-test for PWSprayer2D"""
    sigma = np.zeros((par["nx"], par["nt"]), dtype=par["dtype"])

    Sop = PWSprayer2D(
        dims=(par["nx"], par["nt"]),
        sigma=sigma,
        dtype=par["dtype"],
    )
    dottest(Sop, Sop.shape[0], par["nx"] * par["nt"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PWSmoother2D(par):
    """Dot-test for PWSmoother2D"""
    sigma = np.zeros((par["nx"], par["nt"]), dtype=par["dtype"])

    Sop = PWSmoother2D(
        dims=(par["nx"], par["nt"]),
        sigma=sigma,
        dtype=par["dtype"],
    )
    dottest(Sop, Sop.shape[0], par["nx"] * par["nt"])
