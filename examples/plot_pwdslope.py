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

from pylops.signalprocessing.pwd import PWSmoother2D
from pylops.utils.signalprocessing import pwd_slope_estimate, slope_estimate

plt.close("all")
np.random.seed(10)

###############################################################################
# Load sigmoid model
# ------------------
# Load the synthetic sigmoid model used throughout the notebooks.
sigmoid_path = os.path.join(
    os.path.dirname(__file__), "../testdata/slope_estimate/sigmoid_model.npy"
)
sigmoid = np.load(sigmoid_path)
nz, nx = sigmoid.shape

###############################################################################
# Slope estimation comparison between PWD and Structure Tensor
# ------------------------------------------------------------
# Estimate slopes using both the plane-wave destruction algorithm and the
# structure-tensor estimator. Both return slopes in samples per trace.
pwd_sigma = pwd_slope_estimate(
    sigmoid,
    niter=5,
    liter=20,
    order=2,
    nsmooth=(12, 12),
    damp=0.12,
    smoothing="triangle",
)

st_sigma, _ = slope_estimate(
    sigmoid,
    dz=1.0,
    dx=1.0,
    smooth=5,
    eps=1e-6,
    dips=False,
)
st_sigma = st_sigma.astype(np.float32)

###############################################################################
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

im0 = ax[0].imshow(sigmoid, aspect="auto")
ax[0].set_title("Sigmoid model")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

v = np.max(np.abs(pwd_sigma))
im1 = ax[1].imshow(pwd_sigma, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
ax[1].set_title("PWD slope estimate")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

im2 = ax[2].imshow(st_sigma, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
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

smoother_pwd = PWSmoother2D(
    dims=(nz, nx), sigma=pwd_sigma, radius=radius, alpha=alpha, dtype="float32"
)
smooth_pwd = (smoother_pwd @ noise.ravel()).reshape(nz, nx)

smoother_st = PWSmoother2D(
    dims=(nz, nx), sigma=st_sigma, radius=radius, alpha=alpha, dtype="float32"
)
smooth_st = (smoother_st @ noise.ravel()).reshape(nz, nx)

###############################################################################
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

im0 = ax[0].imshow(noise, aspect="auto", cmap="rainbow")
ax[0].set_title("Random field")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(smooth_pwd, aspect="auto", cmap="rainbow")
ax[1].set_title("Smoothed with PWD slopes")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

im2 = ax[2].imshow(smooth_st, aspect="auto", cmap="rainbow")
ax[2].set_title("Smoothed with structure-tensor slopes")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_xlabel("x (samples)")
ax[0].set_ylabel("z (samples)")
fig.tight_layout()