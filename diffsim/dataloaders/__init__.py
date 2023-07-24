from . larcv import create_larcv_dataset

def build_dataloader(config, MPI_AVAILABLE):
    import pandas as pd
    # Load the sipm database:

    data_args = config.data
    batch_size = config.run.minibatch_size
    batch_keys = ["S2Si", "S2Pmt", "e_deps"]

    input_file = data_args.path
    name       = data_args.name
    distributed = MPI_AVAILABLE
    sparse     = False

    dataset = create_larcv_dataset(
        data_args, batch_size, batch_keys,
        input_file, name,
        distributed, sparse)




    # sipm_db = pd.read_pickle("database/new_sipm.pkl")

    # dl = krypton(
    #     batch_size  = config.run.minibatch_size,
    #     db          = sipm_db,
    #     path        = config.data.path,
    #     run         = config.data.run,
    #     MPI_AVAILABLE=False,
    #     )

    # if config.data.trigger != "":
    #     trigger = config.data.trigger
    # else:
    #     trigger = None

    # k = KryptonLoader(
    #     MPI_AVAILABLE  = MPI_AVAILABLE,
    #     reader_class   = KryptonReader,
    #     path           = config.data.path,
    #     run            = config.data.run,
    #     trigger        = trigger
    # )

    # # Get the next file reader:
    # k.preload_dataset()

    # dl = DataPreloader(
    #     batch_size    = config.run.minibatch_size,
    #     MPI_AVAILABLE = MPI_AVAILABLE,
    #     file_loader   = k
    # )


    return dataset
