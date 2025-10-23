r"""
PWD-based slope estimation and structural smoothing
===================================================

This example shows how to estimate local slopes of a two-dimensional
array using the Plane-Wave Destruction (PWD [1]_) algorithm via
:py:func:`pylops.utils.signalprocessing.pwd_slope_estimate`; such slopes are
then used as a guide to smooth a noise realization following the structural
dips of the input data via the :py:class:`pylops.signalprocessing.PWSmoother2D`.

.. [1] Fomel, S., "Applications of plane‐wave destruction filters",
   Geophysics. 2002.

"""

import matplotlib.pyplot as plt
import numpy as np

import pylops
from pylops.utils.signalprocessing import pwd_slope_estimate, slope_estimate

plt.close("all")
np.random.seed(10)

###############################################################################
# Sigmoid model
# ------------------
# To start we import the same `2d image <http://ahay.org/blog/2014/10/08/program-of-the-month-sfsigmoid/>`_
# that we used in the seislet transform example. This image contains curved
# reflectors that we will try to follow during the smoothing operation.
inputfile = "../testdata/sigmoid.npz"

sigmoid = np.load(inputfile)["sigmoid"].T
nz, nx = sigmoid.shape

###############################################################################
# Slope estimation comparison between PWD and Structure Tensor
# ------------------------------------------------------------
# Next, slopes are estimated using both the plane-wave destruction
# and the structure-tensor algorithms. Both algorithms return slopes
# in samples per trace.
pwd_slope = pwd_slope_estimate(
    sigmoid,
    niter=5,
    liter=20,
    order=2,
    nsmooth=(12, 12),
    damp=6e-4,
    smoothing="triangle",
).astype(np.float32)

st_slope = (-1) * slope_estimate(
    sigmoid,
    dz=1.0,
    dx=1.0,
    smooth=5,
    eps=1e-6,
    dips=False,
)[0].astype(np.float32)

###############################################################################
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

im0 = ax[0].imshow(sigmoid, aspect="auto", cmap="gray")
ax[0].set_title("Sigmoid model")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

v = np.max(np.abs(pwd_slope))
im1 = ax[1].imshow(pwd_slope, aspect="auto", cmap="jet", vmin=-v, vmax=v)
ax[1].set_title("PWD slope estimate")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

im2 = ax[2].imshow(st_slope, aspect="auto", cmap="jet", vmin=-v, vmax=v)
ax[2].set_title("Structure-tensor slope estimate")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_xlabel("x (samples)")
ax[0].set_ylabel("z (samples)")
fig.tight_layout()

###############################################################################
# Structure-aligned smoothing via local slopes
# --------------------------------------------
# The estimated slopes are finally used by the
# :py:class:`pylops.signalprocessing.PWSmoother2D` operator to perform
# structure-aligned smoothing of a random noise realization. Note that
# this operator is defined as the composition of a
# :py:class:`pylops.signalprocessing.PWSprayer2D` operator and its adjoint
# (i.e., ``Sprayer.T @ Sprayer``) and therefore it is
# a symmetric operator.
noise = np.random.uniform(-1.0, 1.0, size=(nz, nx)).astype(np.float32)

radius = 6
alpha = 0.7

SOp_pwd = pylops.signalprocessing.pwd2d.PWSmoother2D(
    dims=(nz, nx), sigma=pwd_slope, radius=radius, alpha=alpha, dtype="float32"
)
smooth_pwd = SOp_pwd @ noise

SOp_st = pylops.signalprocessing.pwd2d.PWSmoother2D(
    dims=(nz, nx), sigma=st_slope, radius=radius, alpha=alpha, dtype="float32"
)
smooth_st = SOp_st @ noise

###############################################################################
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

im0 = ax[0].imshow(noise, aspect="auto", cmap="magma")
ax[0].set_title("Random field")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(smooth_pwd, aspect="auto", cmap="magma")
ax[1].set_title("Smoothed with PWD slopes")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

im2 = ax[2].imshow(smooth_st, aspect="auto", cmap="magma")
ax[2].set_title("Smoothed with structure-tensor slopes")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_xlabel("x (samples)")
ax[0].set_ylabel("z (samples)")
fig.tight_layout()

###############################################################################
# 3D Extension
# ------------
# Finally, we show here how the PWD slope estimation and smoothing
# algorithms can be easily extended to 3D data. We start by generating a 3D synthetic
# volume composed of 10 adjacent 2D sigmoid slices shifted in depth.
ny = 20

sigmoid3d = np.zeros((ny, nz, nx), dtype=sigmoid.dtype)
for i in range(ny):
    sigmoid3d[i, :, :] = np.roll(sigmoid, i / 2, axis=0)

###############################################################################
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, width_ratios=(3, 1))
fig.suptitle("Sigmoid model")
ax[0].imshow(sigmoid3d[ny // 2], aspect="auto", cmap="gray")
ax[1].imshow(sigmoid3d[..., nx // 2].T, aspect="auto", cmap="gray")
fig.tight_layout()

###############################################################################
# Let's now compute the PWD slopes along both ``y`` and ``x`` directions.

pwd_slope3d_y = np.concat(
    [
        pwd_slope_estimate(
            sigmoid3d[i],
            niter=5,
            liter=20,
            order=2,
            nsmooth=(12, 12),
            damp=6e-4,
            smoothing="triangle",
        )[None]
        for i in range(ny)
    ]
).astype(np.float32)

pwd_slope3d_x = (
    np.concat(
        [
            pwd_slope_estimate(
                sigmoid3d[:, :, i].T,
                niter=5,
                liter=20,
                order=2,
                nsmooth=(12, 12),
                damp=6e-4,
                smoothing="triangle",
            )[None]
            for i in range(nx)
        ]
    )
    .transpose(2, 1, 0)
    .astype(np.float32)
)

###############################################################################
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, width_ratios=(3, 1))
fig.suptitle("PWD slopes")
ax[0].imshow(pwd_slope3d_y[ny // 2], aspect="auto", cmap="jet", vmin=-v, vmax=v)
ax[1].imshow(pwd_slope3d_x[..., nx // 2].T, aspect="auto", cmap="jet", vmin=-v, vmax=v)
fig.tight_layout()

###############################################################################
# Let's now compute the PWD slopes along both ``y`` and ``x`` directions.

noise3d = np.random.uniform(-1.0, 1.0, size=(ny, nz, nx)).astype(np.float32)

SOp_pwd3d_y = pylops.BlockDiag(
    [
        pylops.signalprocessing.pwd2d.PWSmoother2D(
            dims=(nz, nx),
            sigma=pwd_slope3d_y[i],
            radius=radius,
            alpha=alpha,
            dtype="float32",
        )
        for i in range(ny)
    ]
)

SOp_pwd3d_x = pylops.BlockDiag(
    [
        pylops.signalprocessing.pwd2d.PWSmoother2D(
            dims=(nz, ny),
            sigma=pwd_slope3d_x[:, :, i].T,
            radius=radius,
            alpha=alpha,
            dtype="float32",
        )
        for i in range(nx)
    ]
)
TOp = pylops.Transpose((ny, nz, nx), axes=(2, 1, 0))
SOp_pwd3d_x = TOp.H @ SOp_pwd3d_x @ TOp

SOp_pwd3d = SOp_pwd3d_x @ SOp_pwd3d_y
smooth_st3d = SOp_pwd3d @ noise3d

###############################################################################
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, width_ratios=(3, 1))
fig.suptitle("Smoothed with structure-tensor slopes")
ax[0].imshow(smooth_st3d[ny // 2], aspect="auto", cmap="magma")
ax[1].imshow(smooth_st3d[..., nx // 2].T, aspect="auto", cmap="magma")
fig.tight_layout()
