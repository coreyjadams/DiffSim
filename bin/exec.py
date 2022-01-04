import sys, os
import pathlib
import time

import signal
import pickle

# For database reads:
import pandas as pd

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


hydra.output_subdir = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"

import tensorflow as tf
# tf.random.set_seed(2)



import logging
from logging import handlers




# Add the local folder to the import path:
src_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(src_dir) + "/src/"
sys.path.insert(0,src_dir)

from config import Config
from config import MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd



class exec(object):

    def __init__(self, config):


        #
        if MPI_AVAILABLE:
            self.rank = hvd.rank()
            self.size = hvd.size()
            self.local_rank = hvd.local_rank()
        else:
            self.rank = 0
            self.size = 1
            self.local_rank = 1

        self.config = config

        self.model_name = self.config.model_name

        self.configure_logger()
        logger = logging.getLogger()
        logger.info("")
        logger.info("\n" + OmegaConf.to_yaml(config))



        # Use this flag to catch interrupts, stop the next step and write output.
        self.active = True

        self.global_step     = 0

        self.save_path  = self.config["save_path"] # Cast to pathlib later

        if "profile" in self.config:
            self.profile = bool(self.config["profile"])
        else:
            self.profile = False


        self.set_compute_parameters()

        # Always construct a dataloader:
        self.dataloader  = self.build_dataloader()

        # If it's just IO testing, run that here then exit:
        from config.mode import ModeKind
        if self.config.mode.name == ModeKind.iotest:
            return


        self.simulator   = self.build_simulator()
        # Run a forward pass once:
        batch = next(self.dataloader.iterate())

        # Run a forward pass but throw away the results:
        self.simulator(batch['energy_deposits'])

        n_parameters = 0
        for p in self.simulator.trainable_variables:
            n_parameters += tf.reduce_prod(p.shape)

        logger.info(f"Number of parameters in this network: {n_parameters}")


        # Only need the res of this if we're training
        if self.config.mode.name != ModeKind.train:
            return

        # These are just for training:
        self.optimizer   = self.build_optimizer()

        self.loss_func   = self.build_loss_function()





        if not MPI_AVAILABLE or hvd.rank() == 0:
            # self.writer = tf.summary.create_file_writer(self.save_path)
            self.writer = tf.summary.create_file_writer(self.save_path + "/log/")


        # Now, cast to pathlib:
        self.save_path = pathlib.Path(self.save_path)


        # We also snapshot the configuration into the log dir:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with open(pathlib.Path('config.snapshot.yaml'), 'w') as cfg:
                OmegaConf.save(config=self.config, f=cfg)

    def configure_logger(self):

        logger = logging.getLogger()

        # Create a handler for STDOUT, but only on the root rank:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 10, target=stream_handler)
            logger.addHandler(handler)
            # Add a file handler:

            # Add a file handler too:
            log_file = self.config.save_path + "/process.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
            logger.addHandler(file_handler)


            logger.setLevel(logging.DEBUG)
            # fh = logging.FileHandler('run.log')
            # fh.setLevel(logging.DEBUG)
            # logger.addHandler(fh)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    def build_dataloader(self):

        from utils.dataloaders import krypton
        # Load the sipm database:
        sipm_db = pd.read_pickle("database/new_sipm.pkl")

        dl = krypton(
            batch_size  = self.config.run.minibatch_size,
            db          = sipm_db,
            path        = self.config.data.path,
            run         = self.config.data.run
            )

        return dl

    def build_simulator(self):
        from simulator.NEW_Simulator import NEXT_Simulator

        simulator = NEXT_Simulator()

        return simulator

    def build_optimizer(self):
        from config.mode import OptimizerKind

        if self.config.mode.optimizer == OptimizerKind.Adam:
            return tf.keras.optimizers.Adam()
        else:
            raise Exception("Unhandled Optimizer")

    def build_loss_function(self):
        from config.mode import Loss

        if self.config.mode.loss == Loss.MSE:
            return tf.keras.losses.MeanSquaredError()
        else:
            raise Exception("Unhandled Loss Kind")


    def restore(self):
        logger = logging.getLogger()

        name = "checkpoint/"
        if not MPI_AVAILABLE or hvd.rank() == 0:
            logger.info("Trying to restore model")


            # Does the model exist?
            # Note that tensorflow adds '.index' and '.data-...' to the name
            tf_p = pathlib.Path(name) / pathlib.Path(str(self.model_name) + ".index")


            # Check for tensorflow first:

            model_restored = False
            tf_found_path = None
            for source_path in [self.save_path, pathlib.Path('./')]:
                if (source_path / tf_p ).is_file():
                    # Note: we use the original path without the '.index' added
                    tf_p = pathlib.Path(name) / pathlib.Path(str(self.model_name))
                    tf_found_path = source_path / tf_p
                    logger.info(f"Resolved weights path is {tf_found_path}")
                    break

            if tf_found_path is None:
                raise OSError(f"{self.model_name} not found.")
            else:
                try:
                    self.simulator.load_weights(tf_found_path)
                    model_restored = True
                    logger.info("Restored from tensorflow!")
                except Exception as e:
                    logger.debug(e)
                    logger.info("Failed to load weights via keras load_weights function.")


            # We get here only if one method restored.
            # Attempt to restore a global step and optimizer but it's not necessary
            try:
                with open(self.save_path / pathlib.Path(name) / pathlib.Path("global_step.pkl"), 'rb') as _f:
                    self.global_step = pickle.load(file=_f)
            except:
                logger.info("Could not restore a global_step or "
                    "an optimizer state.  Starting over with restored weights only.")

    def set_compute_parameters(self):
        # tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)
        tf.debugging.set_log_device_placement(False)
        tf.config.run_functions_eagerly(False)

        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def train(self):

        logger = logging.getLogger()

        #
        # with self.writer.as_default():
        #     tf.summary.graph(self.wavefunction.get_concrete_function().graph)

        # We attempt to restore the weights:
        try:
            self.restore()
            logger.debug("Loaded weights, optimizer and global step!")
        except Exception as excep:
            logger.debug("Failed to load weights!")
            logger.debug(excep)
            pass


        if MPI_AVAILABLE and hvd.size() > 1:
            logger.info("Broadcasting initial model and optimizer state.")
            # We have to broadcast the wavefunction parameter here:
            hvd.broadcast_variables(self.simulator.variables, 0)

            # And the global step:
            self.global_step = hvd.broadcast_object(
                self.global_step, root_rank=0)
            logger.info("Done broadcasting initial model and optimizer state.")


        checkpoint_iteration = 200

        # Before beginning the loop, manually flush the buffer:
        logger.handlers[0].flush()

        best_energy = 999

        dl_iterable = self.dataloader.iterate()

        while self.global_step < self.config.run.iterations:

            if not self.active: break

            if self.profile:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    tf.profiler.experimental.start(str(self.save_path))
                    tf.summary.trace_on(graph=True)

            metrics = {}
            start = time.time()

            batch = next(dl_iterable)

            metrics["io_time"] = time.time() - start

            with tf.GradientTape() as tape:
                generated_s2_image = self.simulator(batch['energy_deposits'])
                # print(generated_s2_image.shape)

                loss = self.loss_func(batch["S2Pmt"],  generated_s2_image)

            metrics['loss'] = loss

            grads = tape.gradient(loss, self.simulator.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.simulator.trainable_variables))

            metrics['time'] = time.time() - start


            self.summary(metrics, self.global_step)

            # # Add the gradients and model weights to the summary every 25 iterations:
            # if self.global_step % 25 == 0:
            #     if not MPI_AVAILABLE or hvd.rank() == 0:
            #         weights = self.sr_worker.wavefunction.trainable_variables
            #         gradients = self.sr_worker.latest_gradients
            #         self.model_summary(weights, gradients, self.global_step)
            #         self.wavefunction_summary(self.sr_worker.latest_psi, self.global_step)


            if self.global_step % 1 == 0:
                logger.info(f"step = {self.global_step}, loss = {metrics['loss'].numpy():.3f}")
                logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io)")

            # Iterate:
            self.global_step += 1

            if self.config.run.checkpoint % self.global_step == 0:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    self.save_weights()
                    pass

            if self.profile:
                if not MPI_AVAILABLE or hvd.rank() == 0:
                    tf.profiler.experimental.stop()
                    tf.summary.trace_off()

        # Save the weights at the very end:
        if not MPI_AVAILABLE or hvd.rank() == 0:
            self.save_weights()

    def analysis(self):

        # in the summary, we make plots and print weights, etc.
        logger = logging.getLogger()

        try:
            self.restore()
            logger.debug("Loaded weights, optimizer and global step!")
        except Exception as excep:
            logger.debug("Failed to load weights!")
            logger.debug(excep)
            pass

        logger.info(self.simulator.trainable_variables)

    def iotest(self):

        logger = logging.getLogger()


        # Before beginning the loop, manually flush the buffer:
        logger.handlers[0].flush()


        dl_iterable = self.dataloader.iterate()

        while self.global_step < self.config.run.iterations:

            if not self.active: break

            metrics = {}
            start = time.time()

            batch = next(dl_iterable)

            metrics["io_time"] = time.time() - start

            metrics['time'] = time.time() - start



            if self.global_step % 1 == 0:
                logger.info(f"step = {self.global_step}")
                logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io)")

            # Iterate:
            self.global_step += 1


    def model_summary(self, weights, gradients, step):
        with self.writer.as_default():
            for w, g in zip(weights, gradients):
                tf.summary.histogram("weights/"   + w.name, w, step=step)
                tf.summary.histogram("gradients/" + w.name, g, step=step)

    def wavefunction_summary(self, latest_psi, step):
        with self.writer.as_default():
            tf.summary.histogram("psi", latest_psi, step=step)


    # @tf.function
    def summary(self, metrics, step):
        if not MPI_AVAILABLE or hvd.rank() == 0:
            with self.writer.as_default():
                for key in metrics:
                    tf.summary.scalar(key, metrics[key], step=step)


    def save_weights(self):

        name = "checkpoint"

        # If the file for the model path already exists, we don't change it until after restoring:
        self.model_path = self.save_path / pathlib.Path(name) / self.model_name


        # Take the network and snapshot it to file:
        self.simulator.save_weights(self.model_path)
        # Save the global step:
        with open(self.save_path /  pathlib.Path(name) / pathlib.Path("global_step.pkl"), 'wb') as _f:
            pickle.dump(self.global_step, file=_f)

    def finalize(self):
        self.dataloader.shutdown()
        
        if not MPI_AVAILABLE or hvd.rank() == 0:
            from config.mode import ModeKind
            if self.config.mode.name == ModeKind.train:
                self.save_weights()

    def interupt_handler(self, sig, frame):
        logger = logging.getLogger()
        self.dataloader.shutdown()
        logger.info("Caught interrupt, exiting gracefully.")
        self.active = False


@hydra.main(config_path="../src/config", config_name="config")
def main(cfg : Config) -> None:


    # Prepare directories:
    work_dir = pathlib.Path(cfg.save_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(cfg.save_path + "/log/")
    log_dir.mkdir(parents=True, exist_ok=True)

    # cd in to the job directory since we disabled that with hydra:
    # os.chdir(cfg.hydra.run.dir)
    e = exec(cfg)
    signal.signal(signal.SIGINT, e.interupt_handler)

    from config.mode import ModeKind
    if cfg.mode.name == ModeKind.iotest:
        e.iotest()
    elif cfg.mode.name == ModeKind.train:
        e.train()
    elif cfg.mode.name == ModeKind.inference:
        e.inference()
    elif cfg.mode.name == ModeKind.analysis:
        e.analysis()
    # elif :
    e.finalize()

if __name__ == "__main__":
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()