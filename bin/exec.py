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
from diffsim.simulator import batch_update_rng_keys

from diffsim.utils import init_mpi, discover_local_rank
from diffsim.utils import summary, model_summary
from diffsim.utils import save_weights, restore_weights
from diffsim.utils import set_compute_parameters, configure_logger, should_do_io

from diffsim.trainers import build_optimizer, close_over_training_step

from diffsim.dataloaders import build_dataloader

from diffsim.utils import comparison_plots

def interupt_handler( sig, frame):
    logger = logging.getLogger()

    logger.info("Finishing iteration and snapshoting weights...")
    global active
    active = False

@jit
def update_summary_params(metrics, sim_params): 

    # Add the diffusion:
    metrics["physics/diffusion_0"] = sim_params["diffusion"]["diff"]["diffusion"][0]
    metrics["physics/diffusion_1"] = sim_params["diffusion"]["diff"]["diffusion"][1]
    metrics["physics/diffusion_2"] = sim_params["diffusion"]["diff"]["diffusion"][2]
    metrics["physics/el_spread"]   = sim_params["el_spread"]["sipm_s2"]["el_spread"][0]
    metrics["physics/lifetime"]    = sim_params["lifetime"]["lifetime"]["lifetime"][0]

    return metrics


@hydra.main(version_base = None, config_path="../diffsim/config/recipes")
def main(cfg : OmegaConf) -> None:

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


    # Create a summary writer:
    if should_do_io(MPI_AVAILABLE, rank):
        writer = SummaryWriter(log_dir, flush_secs=20)
    else:
        writer = None

    # Restore the weights

    try:
        r_w_params, r_opt, r_global_step = restore_weights(cfg.save_path, model_name)
        # Catch the nothing returned case:
        assert r_global_step is not None
        assert r_w_params    is not None
        assert r_opt         is not None
        w_params    = r_w_params
        opt_state   = r_opt
        global_step = r_global_step
        logger.info("Loaded weights, optimizer and global step!")
    except Exception as excep:
        logger.info("Failed to load weights!")
        logger.info(excep)
        pass



    if MPI_AVAILABLE and size > 1:
        logger.info("Broadcasting initial model and opt state.")
        # We have to broadcast the wavefunction parameter here:
        token = None

        # First, flatten the parameter trees:
        w_params_flat, treedef = jax.tree_util.tree_flatten(w_params)

        # need to unfreeze to do this:
        for i, param in enumerate(w_params_flat):
            w_params_flat[i], token = mpi4jax.bcast(
                w_params_flat[i],
                root = 0,
                comm = MPI.COMM_WORLD,
                token = token
            )
        # And re-tree it:
        w_params = jax.tree_util.tree_unflatten(tree_def, w_params_flat)

        # Now do the optimizer the same way:
        opt_state_flat, opt_treedef = jax.tree_util.tree_flatten(opt_state)

        # need to unfreeze to do this:
        for i, param in enumerate(opt_state_flat):
            opt_state_flat[i], token = mpi4jax.bcast(
                opt_state_flat[i],
                root  = 0,
                comm  = MPI.COMM_WORLD,
                token = token
            )
        # And re-tree it:
        opt_state = jax.tree_util.tree_unflatten(opt_treedef, opt_state_flat)


        # And the global step:
        global_step, token = mpi4jax.bcast(global_step,
                        root = 0,
                        comm = MPI.COMM_WORLD,
                        token = token)
        logger.info("Done broadcasting initial model and optimizer state.")

    dl_iterable = dataloader.iterate()
    comp_data = next(dl_iterable)
    # logger.debug("MOVE DATA LOADING BACK")
    
    batch = comp_data
    # # batch = next(dl_iterable)
    # prefactor = {
    #             "S2Pmt" : 1.,
    #             "S2Si"  : 1.
    #         }

    # for key in batch.keys():
    #     if key in prefactor.keys():
    #         batch[key] = prefactor[key]*batch[key]

    global_step = 0

    while global_step < cfg.run.iterations:
        batch = next(dl_iterable)

        if not active: break

        if cfg.run.profile:
            if should_do_io(MPI_AVAILABLE, rank):
                jax.profiler.start_trace(str(cfg.save_path) + "profile")

        metrics = {}
        start = time.time()

        metrics["io_time"] = time.time() - start

        # Split the keys:
        next_rng_keys = batch_update_rng_keys(next_rng_keys)

        sim_params, opt_state, loss, train_metrics = train_step(
            sim_params, 
            opt_state,
            batch,
            next_rng_keys)

        # print(opt_state)

        metrics.update(train_metrics)
        # Add to the metrics the physical parameters:
        metrics = update_summary_params(metrics, sim_params)
        if cfg.run.profile:
            if should_do_io(MPI_AVAILABLE, rank):
                x.block_until_ready()
                jax.profiler.save_device_memory_profile(str(cfg.save_path) + f"memory{global_step}.prof")



        metrics.update({"loss" : loss})


        metrics['time'] = time.time() - start

        if should_do_io(MPI_AVAILABLE, rank):
            summary(writer, metrics, global_step)


        # Add comparison plots every N iterations
        if global_step % cfg.run.image_iteration == 0:
            # print(sim_params)
            if should_do_io(MPI_AVAILABLE, rank):
                save_dir = cfg.save_path / pathlib.Path(f'comp/{global_step}/')

                simulated_data = function_registry["simulate"](
                    sim_params, 
                    comp_data['energy_deposits'], 
                    rngs=next_rng_keys
                )
                comparison_plots(save_dir, simulated_data, comp_data)

        if global_step % 1 == 0:
            logger.info(f"step = {global_step}, loss = {metrics['loss']:.3f}")
            logger.info(f"time = {metrics['time']:.3f} ({metrics['io_time']:.3f} io)")

        # Iterate:
        global_step += 1

        if global_step % cfg.run.checkpoint  == 0:
            if should_do_io(MPI_AVAILABLE, rank):
                save_weights(cfg.save_path, model_name, sim_params, opt_state, global_step)

        if cfg.run.profile:
            if should_do_io(MPI_AVAILABLE, rank):
                jax.profiler.stop_trace()

    # Save the weights at the very end:
    if should_do_io(MPI_AVAILABLE, rank):
        try:
            save_weights(cfg.save_path, model_name, w_params, global_step)
        except:
            pass






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
