r"""
22. Time-shift estimation
=========================
This tutorial showcases how one can leverage the extensive suite of PyLops operators
to solve nonlinear inverse problems with minimal additional boilerplate code. We will
both create a simple nonlinear solver and take advantage of existing solvers provided
by third-party library such as SciPy.

We are going to consider a classic problem in signal processing, namely the registration
of two signals where one signal is a non-stationary shifted version of the other:

.. math::
    d_2(t) = d_1(t - \delta t(t))

where :math:`d_1(t)` and :math:`d_2(t)` are the two signals to register and
:math:`\delta t(t)` is the time-shift that we want to estimate. When :math:`\delta t(t) > 0`,
the second signal is delayed with respect to the first one, whilst when :math:`\delta t(t) < 0`,
the second signal is anticipated with respect to the first one.

"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal.windows import hamming
from scipy.sparse.linalg import aslinearoperator

import pylops

np.random.seed(10)
plt.close("all")

###############################################################################
# Let's first create a signal represented by the superposition of three
# sinusoids and a shifted version of it given a known time-shift function.

# time axis
dt = 0.004
nt = 101
t = np.arange(nt) * dt

# input signal (with taper on the edges)
d1 = (
    np.sin(2 * np.pi * 10 * t)
    + 0.4 * np.sin(2 * np.pi * 20 * t)
    - 2 * np.sin(2 * np.pi * 5 * t)
)
d1 *= hamming(nt)

# define time-shift as integral of a step-like function
steps = np.zeros(nt)
steps[20:70] = -3e-4
shift = np.cumsum(steps)
tshift = t - shift
iava = tshift / dt
SOp, iava = pylops.signalprocessing.Interp(nt, iava, kind="sinc")
d2 = SOp @ d1

# revert time-shift
tshift_rev = t + shift
iava_rev = tshift_rev / dt
SOprev, iava_rev = pylops.signalprocessing.Interp(nt, iava_rev, kind="sinc")
d1back = SOprev @ d2

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].plot(t, shift, "k")
axs[0].set_title("Time-Shift")

axs[1].plot(t, d1, "k", label=r"$d_1(t)$")
axs[1].plot(t, d2, "r", label=r"$d_2(t)=d_1(t - \delta t)$")
axs[1].legend()
axs[1].set_title("Signals")

axs[2].plot(t, d1, "k", label=r"$d_1(t)$")
axs[2].plot(t, d1back, "r", label=r"$d_{1,back}(t)=d_2(t + \delta t)$")
axs[2].legend()
axs[2].set_title("Corrected signal")
fig.tight_layout()

###############################################################################
# We can now try to estimate the time-shift function given the two signals by
# minimizing the following cost function:
#
# .. math::
#      J(\delta t(t)) = ||d_2(t) - d_1(t - \delta t(t))||^2
#
# This is a nonlinear problem as the operator that maps :math:`d_1(t)` into
# :math:`d_1(t - \delta t(t))` depends on the unknown time-shift function
# :math:`\delta t(t)`. We can however solve this problem iteratively by
# linearizing the operator around the current estimate of the time-shift
# function at each iteration. In particular, we can write the Taylor
# expansion of :math:`d_1(t - \delta t(t))` around :math:`t` as:
#
# .. math::
#      d_1(t - \delta t(t)) = d_1(t) - \frac{\partial d_1}{\partial t}|_t \delta t(t)
#
# If we discretize the time axis, we can express this operation in a
# matrix-vector:
#
# .. math::
#      \mathbf{d}_{1, \boldsymbol \delta_\mathbf{t}} = \mathbf{d}_1 + \mathbf{J}
#      \boldsymbol \delta \mathbf{t}
#
# where the Jacobian matrix is given by
# :math:`\mathbf{J}= -diag\{\frac{\partial \mathbf{d}_1}{\partial t}|_{t=t}\}`.
#
# We can now solve the following linear least-squares problem:
#
# .. math::
#      J = ||(\mathbf{d}_2 - \mathbf{d}_1) - \mathbf{J} \boldsymbol
#      \delta \mathbf{t})||_2^2 + \epsilon ||\nabla \boldsymbol \delta \mathbf{t}||_2^2
#
# where a regularization term is added to promote smooth solutions.

# data term
ddiff = d2 - d1

# Jacobian
DOp = pylops.FirstDerivative(nt, sampling=dt, edge=True)
J = -pylops.Diagonal(DOp @ d1)

# second derivative regularization
D2Op = pylops.SecondDerivative(nt)

# inversion
shift_est = pylops.optimization.leastsquares.regularized_inversion(
    J,
    ddiff,
    [
        D2Op,
    ],
    epsRs=[
        1e3,
    ],
    **dict(iter_lim=200)
)[0]

# revert time-shift (with estimated shift)
tshift_est = t + shift_est
iava_est = tshift_est / dt
SOpest, iava_est = pylops.signalprocessing.Interp(nt, iava_est, kind="sinc")
d1back_est = SOpest * d2

fig, axs = plt.subplots(1, 2, figsize=(12, 3))
axs[0].plot(t, shift, "k", label="True")
axs[0].plot(t, shift_est, "r", label="Estimated")
axs[0].set_title("Shifts")

axs[1].plot(t, d1, "k", label=r"$d_1(t)$")
axs[1].plot(t, d1back_est, "r", label=r"$d_{1,back}(t)=d_2(t + \delta \tilde{t})$")
axs[1].plot(t, d1 - d1back_est, "k", lw=0.5)
axs[1].legend()
axs[1].set_title("Corrected signal")
fig.tight_layout()

###############################################################################
# We can see that the estimated time-shift closely matches the true one and
# that the corrected signal is very similar to the original one. However, we
# have so far discarded the higher order terms in the Taylor expansion of
# :math:`d_1(t - \delta t(t))`. We can therefore try to improve our estimate
# by iterating the above procedure a few times, updating the Jacobian at each
# iteration with the current estimate of the time-shift function. In other
# words, at each iteration :math:`i=0,1,...`, we perform the following steps:
#
# - Compute the Jacobian :math:`\mathbf{J}^{i}= -diag\{\frac{\partial
#   \tilde{\mathbf{d}}^i_1}{\partial t}|_{t=t}\}`
# - Solve the linear least-squares problem
#
#   .. math::
#      J = ||(\mathbf{d}_2 - \tilde{\mathbf{d}}^i_1) -  \mathbf{J}^i \boldsymbol
#      \Delta \mathbf{t}^{i+1})||_2^2 + \epsilon ||\nabla (\boldsymbol \Delta \mathbf{t}^{i+1} +
#      \boldsymbol \delta \mathbf{t}^i)||_2^2
#
# - Update the time-shift estimate as
#   :math:`\delta t^{i+1}(t) = \delta t^i(t) + \Delta t^{i+1}(t)`
#   We can repeat these steps until convergence is reached.
# - Time shift :math:`d_1^{i+1}(t)` with the current estimate of the time-shift
#   function: :math:`\tilde{d}_1^{i+1}(t) = d_1^i(t + \delta t^{i+1}(t))`
#
# with :math:`\delta t^0(t)=0` and :math:`\tilde{d}_1^0(t)=d_1(t)`.

# number of outer iterations
niter = 10

# pre-compute derivative operators
Dop = pylops.FirstDerivative(nt, edge=True)
D2Op = pylops.SecondDerivative(nt)

shift_estgn = np.zeros(nt)
shift_estgn_hist = np.zeros((niter, nt))

d1shift = d1.copy()
Jhist_gn = []

for iiter in range(niter):

    # data term
    ddiff = d2 - d1shift

    # compute residual norm
    Jhist_gn.append(np.linalg.norm(ddiff))

    # Jacobian
    J = -pylops.Diagonal((Dop @ d1shift) / dt)

    # inversion
    shift_estgn += pylops.optimization.leastsquares.regularized_inversion(
        J,
        ddiff,
        [
            D2Op,
        ],
        epsRs=[
            5e2,
        ],
        dataregs=[
            -D2Op * shift_estgn,
        ],
        **dict(iter_lim=100, damp=1e-4)
    )[0]
    shift_estgn_hist[iiter] = shift_estgn

    # revert current time-shift estimate
    iava_gn = (t - shift_estgn) / dt
    SOpgn, _ = pylops.signalprocessing.Interp(nt, iava_gn, kind="sinc")
    d1shift = SOpgn @ d1

# compute final residual norm
Jhist_gn.append(np.linalg.norm(d2 - d1shift))

# revert time-shift (with estimated shift)
tshift_est = t + shift_estgn
iava_est = tshift_est / dt
SOpest, iava_est = pylops.signalprocessing.Interp(nt, iava_est, kind="sinc")
d1back_estgn = SOpest * d2

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].plot(t, shift, "k", lw=2, label="True")
axs[0].plot(t, shift_estgn, "r", lw=2, label="Estimated")
axs[0].plot(t, shift_estgn_hist.T, "r", lw=0.5, alpha=0.4)
axs[0].set_title("Shifts")

axs[1].plot(t, d1, "k", label=r"$d_1(t)$")
axs[1].plot(t, d1back_estgn, "r", label=r"$d_{1,back}(t)=d_2(t + \delta \tilde{t})$")
axs[1].plot(t, d1 - d1back_estgn, "k", lw=0.5)
axs[1].legend()
axs[1].set_title("Corrected signal")

axs[2].plot(Jhist_gn, "k")
axs[2].set_title("Residual Norm")
fig.tight_layout()


###############################################################################
# A much better match! However, since we have alternated here the solution of
# linearized systems of equations (for an update in the time-shift) with a
# partial shifting of the input signal :math:`d_1(t)` with the current estimate
# of the time-shift, this pattern makes our solver very be-spoke.
#
# Next, we will see that if we sligthly reformulate our problem in such a way
# that partial shifting is not required, we can take advantage of an existing
# solver provided by a third-party library such as SciPy. To begin with, let's
# rewrite a generic Taylor expansion for :math:`d_1(t - \delta t^{i+1}(t))`
# around :math:`\delta t^i(t)`:
#
# .. math::
#      d_1(t - \delta t^{i+1}(t)) = d_1((t - \delta t^i(t)) -
#      \Delta t^{i+1}(t)) = d_1(t) - \frac{\partial d_1}{\partial t}
#      |_{t=t-\delta t^i(t)}  \Delta t^{i+1}(t)
#
# Again, if we discretize the time axis, we can express this operation in a
# matrix-vector:
#
# .. math::
#      \mathbf{d}_{1, \boldsymbol \delta_\mathbf{t}^{i+1}} = \mathbf{d}^i_1 + \mathbf{J}^i
#      \boldsymbol \Delta \mathbf{t}^{i+1}
#
# where the Jacobian matrix is given by
# :math:`\mathbf{J}^i= -diag\{\frac{\partial \mathbf{d}_1}{\partial t}|_{t=t-\delta t^i(t)}\}`.
#
# By doing so, we can now solve a series of linearized problems of the form:
#
# .. math::
#      J^{i+1} = ||(\mathbf{d}_2 - \tilde{\mathbf{d}}^i_1) - \mathbf{J}^i \boldsymbol
#      \Delta \mathbf{t}^{i+1})||_2^2 + \epsilon ||\nabla (\boldsymbol \Delta \mathbf{t}^{i+1} +
#      \boldsymbol \delta \mathbf{t}^i)||_2^2
#
# where :math:`\tilde{d}_1^i(t) = d_1^i(t + \delta t^i(t))`, :math:`\delta t^0(t)=0`,
# and :math:`\tilde{d}_1^0(t)=d_1(t)`. This series of problems now amenable to the
# `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_
# method. In practice, all we need to be able to create is two methods: the first, called ``fun``, must return
# the inner part of the objective function, the latter, called ``jacobian`` must create a linear operator that
# acts like the Jacobian of the augmented system.


def fun(x, d1, d2, t, dt, eps):
    nt = len(t)
    iava = (t - x) / dt
    SOpest, iava = pylops.signalprocessing.Interp(nt, iava, kind="sinc")
    D2Op = pylops.SecondDerivative(nt)

    d1shift = SOpest * d1
    res = d2 - d1shift
    resr = D2Op * x
    return np.hstack((res, eps * resr))


def jacobian(x, d1, d2, t, dt, eps):
    nt = len(t)
    iava = (t - x) / dt
    SOpest, _ = pylops.signalprocessing.Interp(nt, iava, kind="sinc")
    S1Opest, _ = pylops.signalprocessing.Interp(nt, iava + 1, kind="sinc")
    J = (S1Opest * d1 - SOpest * d1) / dt
    D2Op = pylops.SecondDerivative(nt)
    J = pylops.VStack([pylops.Diagonal(J), eps * D2Op])
    J = aslinearoperator(J)
    return J


def callback(x, t, dt, d1, d2):
    iava = (t - x) / dt
    SOpgn, _ = pylops.signalprocessing.Interp(nt, iava, kind="sinc")
    d1shift = SOpgn @ d1
    shift_estls_hist.append(x)
    Jhist_ls.append(np.linalg.norm(d2 - d1shift))


eps = 8e1
shift_estls_hist = []
Jhist_ls = []
shift_estls = least_squares(
    fun,
    np.zeros(nt),
    jac=jacobian,
    method="trf",
    verbose=1,
    args=(d1, d2, t, dt, eps),
    callback=partial(callback, t=t, dt=dt, d1=d1, d2=d2),
).x

# revert time-shift (with estimated shift)
tshift_est = t + shift_estls
iava_est = tshift_est / dt
SOpest, iava_est = pylops.signalprocessing.Interp(nt, iava_est, kind="sinc")
d1back_estls = SOpest * d2

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].plot(t, shift, "k", lw=2, label="True")
axs[0].plot(t, shift_estls, "r", lw=2, label="Estimated")
axs[0].plot(t, np.vstack(shift_estls_hist).T, "r", lw=0.5, alpha=0.4)
axs[0].set_title("Shifts")

axs[1].plot(t, d1, "k", label=r"$d_1(t)$")
axs[1].plot(t, d1back_estls, "r", label=r"$d_{1,back}(t)=d_2(t + \delta \tilde{t})$")
axs[1].plot(t, d1 - d1back_estls, "k", lw=0.5)
axs[1].legend()
axs[1].set_title("Corrected signal")

axs[2].plot(Jhist_ls, "k")
axs[2].set_title("Residual Norm")
fig.tight_layout()
