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
from pylops.utils import deps, dottest

astra_message = deps.astra_import("the astra module")

if astra_message is None:
    import astra


@pytest.fixture(params=["parallel", "fanflat"])
def proj_geom_type(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def engine(request):
    if request.param == "cuda" and not astra.use_cuda():
        pytest.skip("No CUDA-enabled ASTRA installation found.")
    return request.param


@pytest.fixture
def operator(proj_geom_type, engine):
    nx = 51
    ny = 30
    nthetas = 20
    thetas = np.linspace(0.0, np.pi, nthetas, endpoint=False)
    source_origin_dist = 100 if proj_geom_type == "fanflat" else None
    origin_detector_dist = 0 if proj_geom_type == "fanflat" else None
    return CT2D(
        dims=(ny, nx),
        det_width=1.0,
        det_count=nx,
        thetas=thetas,
        engine=engine,
        proj_geom_type=proj_geom_type,
        source_origin_dist=source_origin_dist,
        origin_detector_dist=origin_detector_dist,
    )


@pytest.fixture
def x(operator):
    return np.ones(operator.dims, dtype=np.float32)


@pytest.fixture
def y(operator):
    return np.ones((len(operator.thetas), operator.det_count), dtype=np.float32)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Not OSX enabled")
class TestCT2D:
    def test_basic(self, operator, x, y):
        assert not np.allclose(operator @ x, 0.0)
        assert not np.allclose(operator.T @ y, 0.0)

    def test_adjointess(self, operator):
        if operator.engine == "cuda":
            pytest.skip("The adjoint with CUDA engine is severely mismatched in ASTRA.")
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
