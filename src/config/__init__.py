from .config import Config

NAME = ""
# NAME = "next-diffsim"

try:
    from mpi4py import MPI # this does MPI init

    MPI_AVAILABLE=True
except BaseException as e:
    MPI_AVAILABLE=False
