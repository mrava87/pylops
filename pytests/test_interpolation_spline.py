import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from pylops.signalprocessing import InterpCubicSpline

TEST_ARRAY_SHAPE = (
    20,
    51,
    2,  # <- this dimension is exactly 2 because this triggers a special solver
    12,
)
TEST_X_RANGE = (-5.0, 5.0)


@pytest.mark.parametrize(
    "with_complex",
    [
        pytest.param(False, id="real"),
        pytest.param(True, id="complex"),
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
    ],
)
@pytest.mark.parametrize(
    "subsample_fraction",
    [
        pytest.param(0.5, id="decimation"),
        pytest.param(5.0, id="upsampling"),
    ],
)
def test_natural_cubic_spline_against_scipy(
    subsample_fraction: float,
    axis: int,
    with_complex: bool,
) -> None:
    """
    Tests ``pylops.signalprocessing.InterpCubicSpline`` against the equivalent
    implementation ``scipy.interpolate.CubicSpline`` for the natural boundary condition.

    """

    # --- Setup ---

    np.random.seed(0)

    num_samples = TEST_ARRAY_SHAPE[axis]
    x_fit = np.linspace(
        start=TEST_X_RANGE[0],
        stop=TEST_X_RANGE[1],
        num=num_samples,
    )
    y_fit = np.random.randn(*TEST_ARRAY_SHAPE) + (
        with_complex * 1.0j
    ) * np.random.randn(*TEST_ARRAY_SHAPE)

    x_eval_fractions = np.random.rand(round(num_samples * subsample_fraction))
    x_eval_for_pylops = (num_samples - 1) * x_eval_fractions
    x_eval_for_scipy = TEST_X_RANGE[0] + x_eval_fractions * (
        TEST_X_RANGE[1] - TEST_X_RANGE[0]
    )

    dtype = "complex128" if with_complex else "float64"

    # --- Test ---

    splinop = InterpCubicSpline(
        dims=TEST_ARRAY_SHAPE,
        iava=x_eval_for_pylops,
        axis=axis,
        dtype=dtype,
    )
    y_eval_pylops = splinop * y_fit

    y_eval_scipy = CubicSpline(
        x=x_fit,
        y=y_fit,
        bc_type="natural",
        axis=axis,
    )(x=x_eval_for_scipy)

    assert np.allclose(y_eval_pylops, y_eval_scipy)

    return
