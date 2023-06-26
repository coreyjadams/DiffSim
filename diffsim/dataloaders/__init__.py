from . krypton import krypton

from . preloader import DataPreloader

from . krypton_threaded import KryptonLoader, KryptonReader

def build_dataloader(config, MPI_AVAILABLE):
    import pandas as pd
    # Load the sipm database:
    sipm_db = pd.read_pickle("database/new_sipm.pkl")

    # dl = krypton(
    #     batch_size  = config.run.minibatch_size,
    #     db          = sipm_db,
    #     path        = config.data.path,
    #     run         = config.data.run,
    #     MPI_AVAILABLE=False,
    #     )


    k = KryptonLoader(
        MPI_AVAILABLE  = MPI_AVAILABLE,
        reader_class   = KryptonReader,
        path           = config.data.path,
        run            = config.data.run,
    )

    # Get the next file reader:
    k.preload_dataset()

    dl = DataPreloader(
        batch_size    = config.run.minibatch_size,
        MPI_AVAILABLE = MPI_AVAILABLE,
        file_loader   = k
    )


    return dl
