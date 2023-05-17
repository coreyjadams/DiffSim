import sys, os
import pathlib
import time

import signal
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# For database reads:
import pandas as pd


import logging
from logging import handlers

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log


hydra.output_subdir = None

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import jax
from jax import random, jit, vmap
import jax.numpy as numpy
import jax.tree_util as tree_util
import optax


from tensorboardX import SummaryWriter


# Add the local folder to the import path:
src_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(src_dir)
sys.path.insert(0,src_dir)

from diffsim.config import Config

# if MPI_AVAILABLE:
#     import horovod.tensorflow as hvd

from diffsim.simulator import init_simulator
# from diffsim.simulator import NEW_Simulator, init_NEW_simulator
from diffsim.simulator import init_rng_keys, update_rng_keys, batch_update_rng_keys

from diffsim.utils import init_mpi, discover_local_rank
from diffsim.utils import summary, model_summary
from diffsim.utils import save_weights, restore_weights
from diffsim.utils import set_compute_parameters, configure_logger, should_do_io

from diffsim.trainers import build_optimizer, close_over_training_step

from diffsim.dataloaders import build_dataloader

def interupt_handler( sig, frame):
    logger = logging.getLogger()

    logger.info("Finishing iteration and snapshoting weights...")
    global active
    active = False

@hydra.main(version_base = None, config_path="../diffsim/config/recipes")
def main(cfg : OmegaConf) -> None:

    print(cfg)
    # Extend the save path:
    # cfg.save_path = cfg.save_path + f"/{cfg.hamiltonian.form}/"
    # cfg.save_path = cfg.save_path + f"/{cfg.sampler.n_particles}particles/"
    # cfg.save_path = cfg.save_path + f"/{cfg.optimizer.solver}_solver/"
    cfg.save_path = cfg.save_path + f"/{cfg.run.id}/"


    # Prepare directories:
    work_dir = pathlib.Path(cfg.save_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(cfg.save_path + "/log/")
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name = pathlib.Path(cfg["model_name"])

    MPI_AVAILABLE, rank, size = init_mpi(cfg.run.distributed)
    
    # Figure out the local rank if MPI is available:
    if MPI_AVAILABLE:
        local_rank = discover_local_rank()
    else:
        local_rank = 0
    # model_name = config.model_name

    configure_logger(log_dir, MPI_AVAILABLE, rank)

    logger = logging.getLogger()
    logger.info("")
    logger.info("\n" + OmegaConf.to_yaml(cfg))



    # Training state variables:
    global active
    active = True

    # Active is a global so that we can interrupt the train loop gracefully
    signal.signal(signal.SIGINT, interupt_handler)

    global_step = 0

    target_device = set_compute_parameters(local_rank)


    

    # Always construct a dataloader:
    dataloader  = build_dataloader(cfg)

    # If it's just IO testing, run that here then exit:
    from diffsim.config.mode import ModeKind
    if cfg.mode.name == ModeKind.iotest:
        iotest(dataloader, cfg)
        return 

    # Initialize the global random seed:
    if cfg.seed == -1:
        global_random_seed = int(time.time())
    else:
        global_random_seed = cfg.seed

    if MPI_AVAILABLE and size > 1:
        if rank == 0:
            # Create a single master key
            master_key = jax.device_put(random.PRNGKey(global_random_seed), target_device)
        else:
            # This isn't meaningful except as a placeholder:
            master_key = jax.device_put(random.PRNGKey(0), target_device)

        # Here, sync up all ranks to the same master key
        import mpi4jax
        from mpi4py import MPI
        master_key, token = mpi4jax.bcast(master_key, root=0, comm=MPI.COMM_WORLD)
    else:
        master_key = jax.device_put(random.PRNGKey(global_random_seed), target_device)

    # Initialize the model:
    example_data = next(dataloader.iterate())
    sim_func, sim_params, next_rng_keys = init_simulator(master_key, cfg, example_data)

    function_registry = {
        "simulate" : sim_func
    }

    n_parameters = 0
    flat_params, tree_def = tree_util.tree_flatten(sim_params)
    for p in flat_params:
        n_parameters += numpy.prod(numpy.asarray(p.shape))
    logger.info(f"Number of parameters in this network: {n_parameters}")


    optimizer, opt_state = build_optimizer(cfg, sim_params)

    function_registry['optimizer'] = optimizer

    #     # self.trainer = self.build_trainer(batch, self.simulator_fn, self.simulator_params)


    train_step = close_over_training_step(function_registry, cfg, MPI_AVAILABLE)

    # def train(self):

    #     logger = logging.getLogger(NAME)

    #     #
    #     # with self.writer.as_default():
    #     #     tf.summary.graph(self.wavefunction.get_concrete_function().graph)

    #     # We attempt to restore the weights:
    #     try:
    #         self.restore()
    #         logger.debug("Loaded weights, optimizer and global step!")
    #     except Exception as excep:
    #         logger.debug("Failed to load weights!")
    #         logger.debug(excep)
    #         pass


    #     if MPI_AVAILABLE and hvd.size() > 1:
    #         logger.info("Broadcasting initial model and optimizer state.")
    #         # We have to broadcast the wavefunction parameter here:
    #         hvd.broadcast_variables(self.simulator.variables, 0)

    #         # And the global step:
    #         self.global_step = hvd.broadcast_object(
    #             self.global_step, root_rank=0)

    #         # And the optimizer:
    #         hvd.broadcast_variables(self.trainer.optimizer.variables(), root_rank=0)
    #         logger.info("Done broadcasting initial model and optimizer state.")


    #     checkpoint_iteration = 200

    #     # Before beginning the loop, manually flush the buffer:
    #     logger.handlers[0].flush()

    #     best_energy = 999

    dl_iterable = dataloader.iterate()
    logger.debug("MOVE DATA LOADING BACK")
    batch = next(dl_iterable)

    global_step = 0

    while global_step < cfg.run.iterations:

        if not active: break

        # if profile:
        #     if not MPI_AVAILABLE or hvd.rank() == 0:
        #         tf.profiler.experimental.start(str(save_path))
        #         tf.summary.trace_on(graph=True)

        metrics = {}
        start = time.time()

        metrics["io_time"] = time.time() - start

        sim_params, opt_state, loss = train_step(
            sim_params, 
            opt_state,
            batch,
            next_rng_keys)

        # UPDATE THE RNG SEEDS!!!

        # model_parameters, train_metrics = \
        #     self.trainer.train_iteration(batch, self.global_step, model_parameters)
        # print(model_parameters.keys())
        # print(model_parameters['diffusion'])
        metrics.update({"loss" : loss})
        # metrics.update(train_metrics)


        metrics['time'] = time.time() - start

        # simulator_metrics = self.simulator.generate_summary_dict()
        # metrics.update(simulator_metrics)


        # self.summary(metrics, self.global_step)

        # # Add comparison plots every iteration for now:
        # if self.global_step % self.cfg.run.image_iteration == 0:
        #     if not MPI_AVAILABLE or hvd.rank() == 0:
        #         save_dir = self.save_path / pathlib.Path(f'comp/{self.global_step}/')
        #         self.trainer.comparison_plots(save_dir)

        # # Add the weights to the summary every 100 iterations:
        # if self.global_step % 100 == 0:
        #     if not MPI_AVAILABLE or hvd.rank() == 0:
        #         parameters = self.trainer.parameters()
        #         # self.model_summary(parameters, self.global_step)


        if global_step % 1 == 0:
            logger.info(f"step = {global_step}, loss = {metrics['loss']:.3f}")
            logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io)")

        # Iterate:
        global_step += 1

        # if self.cfg.run.checkpoint % self.global_step == 0:
        #     if not MPI_AVAILABLE or hvd.rank() == 0:
        #         # TODO here
        #         # self.save_weights()
        #         pass

        # if self.profile:
        #     if not MPI_AVAILABLE or hvd.rank() == 0:
        #         tf.profiler.experimental.stop()
        #         tf.summary.trace_off()

    #TODO HERE
    # # Save the weights at the very end:
    # if not MPI_AVAILABLE or hvd.rank() == 0:
    #     self.save_weights()







    # def build_trainer(self, batch, fn, params):

    #     # Shouldn't reach this portion unless training.
    #     from trainers import supervised_trainer

    #     trainer = supervised_trainer(self.config.mode, batch, fn, params)
    #     return trainer

    # def restore(self):
    #     logger = logging.getLogger(NAME)

    #     name = "checkpoint/"
    #     if not MPI_AVAILABLE or hvd.rank() == 0:
    #         logger.info("Trying to restore model")


    #         # Does the model exist?
    #         # Note that tensorflow adds '.index' and '.data-...' to the name
    #         tf_p = pathlib.Path(name) / pathlib.Path(str(self.model_name) + ".index")


    #         # Check for tensorflow first:

    #         model_restored = False
    #         tf_found_path = None
    #         for source_path in [self.save_path, pathlib.Path('./')]:
    #             if (source_path / tf_p ).is_file():
    #                 # Note: we use the original path without the '.index' added
    #                 tf_p = pathlib.Path(name) / pathlib.Path(str(self.model_name))
    #                 tf_found_path = source_path / tf_p
    #                 logger.info(f"Resolved weights path is {tf_found_path}")
    #                 break

    #         if tf_found_path is None:
    #             raise OSError(f"{self.model_name} not found.")
    #         else:
    #             try:
    #                 self.simulator.load_weights(tf_found_path)
    #                 model_restored = True
    #                 logger.info("Restored from tensorflow!")
    #             except Exception as e:
    #                 logger.debug(e)
    #                 logger.info("Failed to load weights via keras load_weights function.")


    #         # We get here only if one method restored.
    #         # Attempt to restore a global step and optimizer but it's not necessary
    #         try:
    #             with open(self.save_path / pathlib.Path(name) / pathlib.Path("global_step.pkl"), 'rb') as _f:
    #                 self.global_step = pickle.load(file=_f)
    #         except:
    #             logger.info("Could not restore a global_step or "
    #                 "an optimizer state.  Starting over with restored weights only.")


    # def analysis(self):

    #     # in the summary, we make plots and print weights, etc.
    #     logger = logging.getLogger(NAME)

    #     try:
    #         self.restore()
    #         logger.debug("Loaded weights, optimizer and global step!")
    #     except Exception as excep:
    #         logger.debug("Failed to load weights!")
    #         logger.debug(excep)
    #         pass

    #     logger.info(self.simulator.trainable_variables)

def iotest(dataloader, config):

    logger = logging.getLogger()


    # Before beginning the loop, manually flush the buffer:
    logger.handlers[0].flush()

    global active
    active = True

    dl_iterable = dataloader.iterate()

    global_step = 0

    while global_step < config.run.iterations:

        if not active: break

        metrics = {}
        start = time.time()

        batch = next(dl_iterable)

        metrics["io_time"] = time.time() - start

        metrics['time'] = time.time() - start



        if global_step % 1 == 0:
            logger.info(f"step = {global_step}")
            logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io)")

        # Iterate:
        global_step += 1


    # def model_summary(self, weights, step):
    #     # with self.writer.as_default():
    #     for key in weights:
    #         self.writer.add_histogram("weights/" + key, weights[key], step)

    # def wavefunction_summary(self, latest_psi, step):
    #     # with self.writer.as_default():
    #     self.writer.add_histogram("psi", latest_psi, step)


    # # @tf.function
    # def summary(self, metrics, step):
    #     if not MPI_AVAILABLE or hvd.rank() == 0:
    #         # with self.writer.as_default():
    #         for key in metrics:
    #             self.writer.add_scalar(key, metrics[key], step)


    # def save_weights(self):

    #     name = "checkpoint"

    #     # If the file for the model path already exists, we don't change it until after restoring:
    #     self.model_path = self.save_path / pathlib.Path(name) / self.model_name


    #     # Take the network and snapshot it to file:
    #     self.simulator.save_weights(self.model_path)
    #     # Save the global step:
    #     with open(self.save_path /  pathlib.Path(name) / pathlib.Path("global_step.pkl"), 'wb') as _f:
    #         pickle.dump(self.global_step, file=_f)

    # def finalize(self):
    #     self.dataloader.shutdown()

    #     if not MPI_AVAILABLE or hvd.rank() == 0:
    #         from config.mode import ModeKind
    #         if self.config.mode.name == ModeKind.train:
    #             # self.save_weights()
    #             pass

    # def interupt_handler(self, sig, frame):
    #     logger = logging.getLogger(NAME)
    #     self.dataloader.shutdown()
    #     logger.info("Caught interrupt, exiting gracefully.")
    #     self.active = False


# @hydra.main(config_path="../src/config", config_name="config")
# def main(cfg : Config) -> None:

#     # Prepare directories:
#     work_dir = pathlib.Path(cfg.save_path)
#     work_dir.mkdir(parents=True, exist_ok=True)
#     log_dir = pathlib.Path(cfg.save_path + "/log/")
#     log_dir.mkdir(parents=True, exist_ok=True)

#     # cd in to the job directory since we disabled that with hydra:
#     # os.chdir(cfg.hydra.run.dir)
#     e = exec(cfg)
#     signal.signal(signal.SIGINT, e.interupt_handler)

#     from config.mode import ModeKind
#     if cfg.mode.name == ModeKind.iotest:
#         e.iotest()
#     elif cfg.mode.name == ModeKind.train:
#         e.train()
#     elif cfg.mode.name == ModeKind.inference:
#         e.inference()
#     elif cfg.mode.name == ModeKind.analysis:
#         e.analysis()
#     # elif :
#     e.finalize()

if __name__ == "__main__":
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += [
            'hydra/job_logging=disabled',
            'hydra.output_subdir=null',
            'hydra.job.chdir=False',
            'hydra.run.dir=.',
            'hydra/hydra_logging=disabled',
        ]


    main()
