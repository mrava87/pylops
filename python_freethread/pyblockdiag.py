import sys
import time
import numpy as np
from pylops.basicoperators.blockdiag_MULTITHREAD import BlockDiag
from pylops.avo import PoststackLinearModelling
from pylops.utils.wavelets import ricker
from threadpoolctl import threadpool_limits


def main():
    nxgroup, nt = 8000, 4000
    ngroups = 8
    workers = 2
    explicit = False
    
    # wavelet
    ntwav = 41
    t0 = np.arange(ntwav) * 0.005
    wav, _, _ = ricker(t0, 20)

    POp = PoststackLinearModelling(
        wav / 2, nt0=nt, spatdims=(nxgroup * ngroups, ), explicit=explicit)
    
    # Baseline
    x = np.ones(POp.shape[1])
    
    t0 = time.perf_counter()
    y = POp @ x
    tend = time.perf_counter()
    print(f"Forward time (baseline): {tend - t0:.2f}s")

    t0 = time.perf_counter()
    x1 = POp.H @ y
    tend = time.perf_counter()
    print(f"Adj time (baseline): {tend - t0:.2f}s")

    # Multithreaded
    Ops = []
    for i in range(ngroups):
        Op = PoststackLinearModelling(
            wav / 2, nt0=nt, spatdims=(nxgroup, ), explicit=explicit)
        Ops.append(Op.H)
    BOp = BlockDiag(Ops, nproc=workers, multiproc=False)
    print("BOp.workers", BOp.nproc)
    print("BOp.multiproc", BOp.multiproc)
    BOp = BOp.H
    print("BOp threaded", BOp)

    t0 = time.perf_counter()
    with threadpool_limits(limits=4 // workers):
        y = BOp @ x
    tend = time.perf_counter()
    print(f"Forward time (baseline): {tend - t0:.2f}s")

    t0 = time.perf_counter()
    with threadpool_limits(limits=2):
        x1 = BOp.H @ y
    tend = time.perf_counter()
    print(f"Adj time (baseline): {tend - t0:.2f}s")

if __name__ == "__main__":
    print("Use GIL:", sys._is_gil_enabled())
    main()
