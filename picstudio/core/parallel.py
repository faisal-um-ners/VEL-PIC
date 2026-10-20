"""Parallel execution helpers.

* threads : shared-memory parallelism of the numba kernels (prange over
            particle chunks with private deposit buffers).
* mpi     : particle decomposition with mpi4py.  Every rank owns a subset of
            the macro-particles and a full copy of the (2D) mesh; charge and
            current densities are summed with MPI_Allreduce and every rank
            performs the (cheap) field solve.  Launch with
                mpiexec -n 4 python -m picstudio run model.pic.json
"""
from __future__ import annotations

import os

import numpy as np


class SerialComm:
    rank = 0
    size = 1
    is_mpi = False

    def allreduce(self, a):
        return a

    def allsum(self, x):
        return x

    def bcast(self, x):
        return x

    def barrier(self):
        pass


class MPIComm:
    is_mpi = True

    def __init__(self):
        from mpi4py import MPI
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def allreduce(self, a):
        a = np.ascontiguousarray(a)
        out = np.empty_like(a)
        self.comm.Allreduce(a, out, op=self.MPI.SUM)
        return out

    def allsum(self, x):
        return self.comm.allreduce(x, op=self.MPI.SUM)

    def bcast(self, x):
        return self.comm.bcast(x, root=0)

    def barrier(self):
        self.comm.Barrier()


def mpi_launched() -> bool:
    keys = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MPI_LOCALNRANKS", "MV2_COMM_WORLD_SIZE")
    return any(k in os.environ for k in keys)


def get_comm(mode: str):
    if mode == "mpi" or mpi_launched():
        try:
            c = MPIComm()
            if c.size > 1 or mode == "mpi":
                return c
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: MPI requested but mpi4py/MPI is not usable ({str(exc).splitlines()[0]}); "
                  f"running without MPI.", flush=True)
    return SerialComm()


def configure_threads(mode: str, nthreads: int, comm=None) -> int:
    """Set the numba thread count and return the number of deposit chunks."""
    import numba
    maxt = numba.config.NUMBA_NUM_THREADS
    if mode == "serial":
        n = 1
    else:
        n = nthreads if nthreads and nthreads > 0 else maxt
        if comm is not None and comm.size > 1 and not nthreads:
            n = max(1, maxt // comm.size)
    n = max(1, min(n, maxt))
    if comm is not None and comm.size > 1:
        import os
        cores = os.cpu_count() or maxt
        if n * comm.size > cores:          # avoid oversubscription (ranks x threads > cores)
            n2 = max(1, cores // comm.size)
            if comm.rank == 0:
                print(f"NOTE: {comm.size} MPI ranks x {n} threads > {cores} cores; using {n2} thread(s) "
                      f"per rank", flush=True)
            n = n2
    numba.set_num_threads(n)
    return n
