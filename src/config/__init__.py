from .config import Config

try:
    from mpi4py import MPI # this does MPI init
    import horovod.tensorflow as hvd
    hvd.init()

    # This is to force each rank onto it's own GPU:
    if (hvd.size() != 1 ):
        import tensorflow as tf
        # Only set this if there is more than one GPU.  Otherwise, its probably
        # Set elsewhere
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if hvd and len(gpus) > 0:
            tf.config.set_visible_devices(gpus[hvd.local_rank() % len(gpus)],'GPU')
    MPI_AVAILABLE=True
except BaseException as e:
    MPI_AVAILABLE=False
