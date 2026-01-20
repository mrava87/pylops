import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal, assert_array_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal, assert_array_equal

    backend = "numpy"
import pytest

from pylops.medical import MRI2D
from pylops.utils import dottest, mkl_fft_enabled

par1 = {
    "ny": 32,
    "nx": 64,
    "imag": 0,
    "dtype": "complex128",
    "engine": "numpy",
}  # real input, complex dtype, numpy engine
par2 = {
    "ny": 32,
    "nx": 64,
    "imag": 1j,
    "dtype": "complex128",
    "engine": "numpy",
}  # complex input, complex dtype, numpy engine
par3 = {
    "ny": 32,
    "nx": 64,
    "imag": 0,
    "dtype": "complex64",
    "engine": "numpy",
}  # real input, complex64 dtype, numpy engine
par4 = {
    "ny": 32,
    "nx": 64,
    "imag": 1j,
    "dtype": "complex64",
    "engine": "scipy",
}  # complex input, complex64 dtype, scipy engine

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_MRI2D_mask_array(par):
    """Dot-test and forward/adjoint for MRI2D operator with numpy array mask"""
    np.random.seed(10)

    # Create a random mask
    mask = np.zeros((par["ny"], par["nx"]), dtype=bool)
    nselected = int(par["ny"] * par["nx"] * 0.3)
    indices = np.random.choice(par["ny"] * par["nx"], nselected, replace=False)
    mask.flat[indices] = True
    mask = mask.astype(par["dtype"])

    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask=mask,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    assert dottest(
        Mop,
        nselected,
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == nselected
    assert xadj.shape[0] == par["ny"] * par["nx"]


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_vertical_reg(par):
    """Dot-test and forward/adjoint for MRI2D operator with vertical-reg mask"""
    np.random.seed(10)

    nlines = 16
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-reg",
        nlines=nlines,
        perc_center=0.0,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    assert dottest(
        Mop,
        par["ny"] * nlines,
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == par["ny"] * nlines
    assert xadj.shape[0] == par["ny"] * par["nx"]
    assert len(Mop.mask) == nlines


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_vertical_uni(par):
    """Dot-test and forward/adjoint for MRI2D operator with vertical-uni mask"""
    np.random.seed(10)

    nlines = 16
    perc_center = 0.1
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-uni",
        nlines=nlines,
        perc_center=perc_center,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    assert dottest(
        Mop,
        par["ny"] * (nlines + int(perc_center * par["nx"])),
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    y = Mop * x.ravel()
    xadj = Mop.H * y

    nlines_total = nlines + int(perc_center * par["nx"])
    assert y.shape[0] == par["ny"] * nlines_total
    assert xadj.shape[0] == par["ny"] * par["nx"]
    assert len(Mop.mask) == nlines_total


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_radial_reg(par):
    """Dot-test and forward/adjoint for MRI2D operator with radial-reg mask"""
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="radial-reg",
        nlines=nlines,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    # For radial masks, output size depends on the number of points in the mask
    npoints = Mop.mask.shape[1]

    assert dottest(
        Mop,
        npoints,
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == npoints
    assert xadj.shape[0] == par["ny"] * par["nx"]
    assert Mop.mask.shape[0] == 2  # x and y coordinates


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_radial_uni(par):
    """Dot-test and forward/adjoint for MRI2D operator with radial-uni mask"""
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="radial-uni",
        nlines=nlines,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    # For radial masks, output size depends on the number of points in the mask
    npoints = Mop.mask.shape[1]

    assert dottest(
        Mop,
        npoints,
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == npoints
    assert xadj.shape[0] == par["ny"] * par["nx"]
    assert Mop.mask.shape[0] == 2  # x and y coordinates


def test_MRI2D_invalid_mask():
    """Test MRI2D operator with invalid mask string"""
    with pytest.raises(ValueError, match="mask must be"):
        MRI2D(
            dims=(32, 64),
            mask="invalid-mask",
            nlines=16,
            dtype="complex128",
        )


def test_MRI2D_invalid_engine():
    """Test MRI2D operator with invalid engine"""
    mask = np.zeros((32, 64), dtype="complex128")
    mask[::2, ::2] = 1.0

    with pytest.raises(ValueError, match="engine must be"):
        MRI2D(
            dims=(32, 64),
            mask=mask,
            engine="invalid-engine",
            dtype="complex128",
        )


def test_MRI2D_vertical_reg_invalid_perc_center():
    """Test MRI2D operator with vertical-reg mask and non-zero perc_center"""
    with pytest.raises(ValueError, match="perc_center must be 0.0"):
        MRI2D(
            dims=(32, 64),
            mask="vertical-reg",
            nlines=16,
            perc_center=0.1,
            dtype="complex128",
        )


def test_MRI2D_vertical_mask_invalid_nlines():
    """Test MRI2D operator with vertical mask and invalid nlines"""
    with pytest.raises(ValueError, match="nlines and perc_center"):
        MRI2D(
            dims=(32, 64),
            mask="vertical-uni",
            nlines=60,
            perc_center=0.5,
            dtype="complex128",
        )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_vertical_mask_no_center(par):
    """Test MRI2D operator with vertical mask and no center lines"""
    np.random.seed(10)

    nlines = 16
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-uni",
        nlines=nlines,
        perc_center=0.0,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    assert len(Mop.mask) == nlines
    assert dottest(
        Mop,
        par["ny"] * nlines,
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_attributes(par):
    """Test MRI2D operator attributes"""
    np.random.seed(10)

    mask = np.zeros((par["ny"], par["nx"]), dtype=bool)
    mask[::2, ::2] = True
    mask = mask.astype(par["dtype"])

    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask=mask,
        engine=par["engine"],
        dtype=par["dtype"],
        name="TestMRI",
    )

    assert Mop.dims == (par["ny"], par["nx"])
    assert Mop.engine == par["engine"]
    assert Mop.name == "TestMRI"
    assert hasattr(Mop, "mask")
    assert hasattr(Mop, "ROp")
    assert Mop.shape[1] == par["ny"] * par["nx"]


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_non_contiguous_input(par):
    """Test MRI2D operator with non-contiguous input"""
    np.random.seed(10)

    mask = np.zeros((par["ny"], par["nx"]), dtype=bool)
    mask[::2, ::2] = True
    mask = mask.astype(par["dtype"])

    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask=mask,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    x_noncontig = x[:, ::-1]  # non-contiguous view

    y = Mop * x_noncontig.ravel()
    assert not np.allclose(y, 0.0)


@pytest.mark.skipif(not mkl_fft_enabled(), reason="MKL FFT not available")
@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_mkl_engine(par):
    """Test MRI2D operator with MKL FFT engine"""
    np.random.seed(10)

    mask = np.zeros((par["ny"], par["nx"]), dtype=bool)
    mask[::2, ::2] = True
    mask = mask.astype(par["dtype"])

    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask=mask,
        engine="mkl_fft",
        dtype=par["dtype"],
    )

    assert dottest(
        Mop,
        np.sum(mask),
        par["ny"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        backend=backend,
    )

    x = np.random.normal(0, 1, (par["ny"], par["nx"])) + par["imag"] * np.random.normal(
        0, 1, (par["ny"], par["nx"])
    )
    y = Mop * x.ravel()
    xadj = Mop.H * y

    assert y.shape[0] == np.sum(mask)
    assert xadj.shape[0] == par["ny"] * par["nx"]


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_vertical_mask_regularity(par):
    """Test that vertical-reg mask produces regularly spaced lines"""
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="vertical-reg",
        nlines=nlines,
        perc_center=0.0,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    mask_indices = Mop.mask
    # Check that indices are regularly spaced
    if len(mask_indices) > 1:
        steps = np.diff(np.sort(mask_indices))
        # All steps should be approximately equal (within rounding)
        assert np.allclose(steps, steps[0], atol=1)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MRI2D_radial_mask_shape(par):
    """Test that radial mask has correct shape"""
    np.random.seed(10)

    nlines = 8
    Mop = MRI2D(
        dims=(par["ny"], par["nx"]),
        mask="radial-reg",
        nlines=nlines,
        engine=par["engine"],
        dtype=par["dtype"],
    )

    # Radial mask should be 2 x npoints array
    assert Mop.mask.shape[0] == 2
    assert Mop.mask.shape[1] > 0
    # All points should be within bounds
    assert np.all(Mop.mask[0] >= 0)
    assert np.all(Mop.mask[0] < par["ny"])
    assert np.all(Mop.mask[1] >= 0)
    assert np.all(Mop.mask[1] < par["nx"])
