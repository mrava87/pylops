__all__ = ["scalability_test"]

import logging
import time
from typing import List, Optional, Tuple

import numpy.typing as npt

logger = logging.getLogger(__name__)


def scalability_test(
    Op,
    x: npt.ArrayLike,
    workers: Optional[List[int]] = None,
    forward: bool = True,
    ntimes: int = 1,
) -> Tuple[List[float], List[float]]:
    r"""Scalability test.

    Small auxiliary routine to test the performance of operators using
    ``multiprocessing``/``concurrent.futures``. This helps identifying the
    maximum number of workers beyond which no performance gain is observed.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to test. It must allow for multiprocessing/multithreading.
    x : :obj:`numpy.ndarray`, optional
        Input vector.
    workers : :obj:`list`, optional
        Number of workers to test out. Defaults to `[1, 2, 4]`.
    forward : :obj:`bool`, optional
        Apply forward (``True``) or adjoint (``False``)
    ntimes : :obj:`int`, optional
        Number of times the forward/adjoint is applied whilst timing
        operations. Consider using :math:`n_{times} \ge 10` to obtain
        a stable measure of the compute times and speedups.

    Returns
    -------
    compute_times : :obj:`list`
        Compute times as function of workers
    speedup : :obj:`list`
        Speedup as function of workers

    """
    if workers is None:
        workers = [1, 2, 4]
    compute_times = []
    speedup = []
    for nworkers in workers:
        logger.info("Working with %d workers...", nworkers)
        # update number of workers in operator
        Op.nproc = nworkers
        # run forward/adjoint computation
        starttime = time.perf_counter()
        for _ in range(ntimes):
            if forward:
                _ = Op.matvec(x)
            else:
                _ = Op.rmatvec(x)
        elapsedtime = (time.perf_counter() - starttime) / ntimes
        compute_times.append(elapsedtime)
        speedup.append(compute_times[0] / elapsedtime)
    Op.close()
    return compute_times, speedup
