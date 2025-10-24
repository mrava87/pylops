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


@pytest.fixture
def operator(request):
    geometry_type = request.param
    nx = 51
    ny = 30
    ntheta = 20
    theta = np.linspace(0.0, np.pi, ntheta, endpoint=False)
    source_origin_dist = 100 if geometry_type == "fanflat" else None
    origin_detector_dist = 0 if geometry_type == "fanflat" else None
    return CT2D(
        (ny, nx),
        1.0,
        ny,
        theta,
        geometry_type,
        source_origin_dist,
        origin_detector_dist,
        engine="cpu" if backend == "numpy" else "cuda",
    )


@pytest.fixture
def x(operator):
    return np.ones(operator.dims, dtype=np.float32)


@pytest.fixture
def y(operator):
    return np.ones((len(operator.thetas), operator.det_count), dtype=np.float32)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
@pytest.mark.parametrize("operator", ["parallel", "fanflat"], indirect=True)
class TestCT2D:
    def test_basic(self, operator, x, y):
        assert not np.allclose(operator @ x, 0.0)
        assert not np.allclose(operator.T @ y, 0.0)

    @pytest.mark.skipif(
        backend == "cupy",
        reason="CUDA tests are failing because of severely mismatched adjoint.",
    )
    def test_adjointess(self, operator):
        assert dottest(operator, rtol=1e-5)

    def test_non_astra_native_dtype(self, operator, x, y):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        assert not np.allclose(operator @ x, 0.0)
        assert not np.allclose(operator.T @ y, 0.0)

    def test_non_contiguous_input(self, operator, x, y):
        x = x[:, ::-1]  # non-contiguous view
        y = y[:, ::-1]
        assert not np.allclose(operator @ x, 0.0)
        assert not np.allclose(operator.T @ y, 0.0)
