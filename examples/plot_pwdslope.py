r"""
Structural smoothing and slope estimation via Plane-Wave Destructors
====================================================================

We compare PWD-based slope estimation against the
structure-tensor algorithm and show how the structural smoother
smooths the random noise along structural dips.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from pylops.signalprocessing.pwd_spraying2d import PWSmoother2D
from pylops.utils.signalprocessing import pwd_slope_estimate, slope_estimate

plt.close("all")
np.random.seed(10)

###############################################################################
# Load sigmoid model
# ------------------
# Load the synthetic sigmoid model used throughout the notebooks.
inputfile = "../testdata/sigmoid.npz"
sigmoid = np.load(inputfile)["sigmoid"].T
nz, nx = sigmoid.shape

###############################################################################
# Slope estimation comparison between PWD and Structure Tensor
# ------------------------------------------------------------
# Estimate slopes using both the plane-wave destruction algorithm and the
# structure-tensor estimator. Both return slopes in samples per trace.
pwd_slope = pwd_slope_estimate(
    sigmoid,
    niter=5,
    liter=20,
    order=2,
    nsmooth=(12, 12),
    damp=6e-4,
    smoothing="triangle",
)

st_slope = (-1) * slope_estimate(
    sigmoid,
    dz=1.0,
    dx=1.0,
    smooth=5,
    eps=1e-6,
    dips=False,
)[0]
st_slope = st_slope.astype(np.float32)

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
# Build the PWSmoother2D operator (Sprayer.T @ Sprayer) from both slope fields
# and highlight the effect of structure-aligned smoothing.
noise = np.random.uniform(-1.0, 1.0, size=(nz, nx)).astype(np.float32)

radius = 6
alpha = 0.7

SOp_pwd = PWSmoother2D(
    dims=(nz, nx), sigma=pwd_slope, radius=radius, alpha=alpha, dtype="float32"
)
smooth_pwd = SOp_pwd @ noise

SOp_st = PWSmoother2D(
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
