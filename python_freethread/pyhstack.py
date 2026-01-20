import sys
import time
import numpy as np
from pylops.basicoperators.vstack_MULTITHREAD import VStack
from pylops.waveeqprocessing import BlendingContinuous
from threadpoolctl import threadpool_limits


def main():
    ns, nr, nt = 800, 200, 1000
    nsgroups = 8
    workers = 2
    
    overlap = 0.5
    dt = 0.004
    ignition_times = 2.0 * np.random.rand(ns) - 1.0
    ignition_times = np.arange(0, overlap * nt * ns, overlap * nt) * dt + ignition_times
    ignition_times[0] = 0.
    
    ns_per_group = ns // nsgroups
    ignition_times_max = ignition_times.max()
    for i in range(0, ns, ns_per_group):
        ignition_times[i + ns_per_group - 1] = ignition_times_max
    
    BOp = BlendingContinuous(nt, nr, ns, dt, ignition_times, shiftall=True, dtype="complex128")
    print("BOp serial", BOp)
    
    # Baseline
    x = np.ones(BOp.shape[1])
    
    t0 = time.perf_counter()
    y = BOp @ x
    tend = time.perf_counter()
    print(f"Forward time (baseline): {tend - t0:.2f}s")

    t0 = time.perf_counter()
    x1 = BOp.H @ y
    tend = time.perf_counter()
    print(f"Adj time (baseline): {tend - t0:.2f}s")

    # Multithreaded
    ns_per_group = ns // nsgroups
    is_start = np.arange(0, ns - ns_per_group + 1, ns_per_group)
    is_end = is_start + ns_per_group

    Ops = []
    for i in range(nsgroups):
        Op = BlendingContinuous(nt, nr, ns_per_group, dt, 
                                 ignition_times[is_start[i]:is_end[i]],
                                 shiftall=True, dtype="complex128")
        Ops.append(Op.H)
    BOp = VStack(Ops, nproc=workers, multiproc=False)
    print("BOp.workers", BOp.nproc)
    print("BOp.multiproc", BOp.multiproc)
    BOp = BOp.H
    print("BOp threaded", BOp)

    t0 = time.perf_counter()
    with threadpool_limits(limits=2):
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
