from . krypton import krypton

def build_dataloader(config):
    import pandas as pd
    # Load the sipm database:
    sipm_db = pd.read_pickle("database/new_sipm.pkl")

    dl = krypton(
        batch_size  = config.run.minibatch_size,
        db          = sipm_db,
        path        = config.data.path,
        run         = config.data.run,
        MPI_AVAILABLE=False,
        )

    return dl