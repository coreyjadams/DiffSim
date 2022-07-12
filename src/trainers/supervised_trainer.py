import time

from config import MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd

import jax
import jax.numpy as numpy
import jax.tree_util as tree_util
from jax import jit

from jax.experimental import optimizers as jax_opt

import logging
from logging import handlers

logger = logging.getLogger()

import pathlib
from matplotlib import pyplot as plt

# Define some easily jit'd functions for loss and gradient applications:
@jit
def compute_loss(simulated_waveforms, real_waveforms):
    # First, we compute the difference squared in the two:
    difference = (simulated_waveforms - real_waveforms)**2.
    # Because this data is so sparse, we multiply the squared difference
    # by the input data to push the loss up in important places and
    # down in unimportant places.  But, don't want the loss to be zero 
    # where the wavefunction is zero, so put a floor:
    loss = difference
    # loss = difference * (real_waveforms + 1e-4)
    
    # And, return the loss as a scalar:
    return loss.mean()




@jit
def apply_gradients(_parameters, _gradients, learning_rate):
    
    # Flatten the tree:
    p_flat_tree, p_tree_def = tree_util.tree_flatten(_parameters)
    g_flat_tree, g_tree_def = tree_util.tree_flatten(_gradients)
    
    new_params = [p - learning_rate * g for p, g in zip(p_flat_tree, g_flat_tree)]

    return tree_util.tree_unflatten(p_tree_def, new_params)


class supervised_trainer:

    def __init__(self, config, monitor_data, simulate_fn, parameters):

        self.config = config

        # These are just for training:
        # self.optimizer   = self.build_optimizer()

        # self.loss_func   = self.build_loss_function()

        # Hold one batch of data to make plots while training.
        self.monitor_data = monitor_data

        self.simulate_fn = simulate_fn

        self.key = jax.random.PRNGKey(int(time.time()))
        self.key, subkey = jax.random.split(self.key)

        # Creat a local forward pass function to use to create a grad function:

        @jit
        def forward_pass(batch, _parameters, key):
            simulated_pmts = simulate_fn(batch['energy_deposits'], _parameters, key)
            loss = compute_loss(simulated_pmts, batch['S2Pmt'])
            return loss, simulated_pmts

        self.gradient_fn = jit(jax.value_and_grad(forward_pass, argnums=1, has_aux=True))





        opt_init, opt_update, get_params = jax_opt.adam(1e-3)
        
        # Initialize the optimizer:
        self.opt_state = opt_init(parameters)



        self.opt_update = opt_update
        self.get_params = get_params




    def plot_pmts(self, plot_dir, sim_pmts, real_pmts):

        # x_ticks = numpy.arange(550)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for i_pmt in range(12):

            # Find the peak of this PMT and only plot the nearby data:
            peak_tick = real_pmts[i_pmt].argmax()

            start = max(peak_tick - 50, 0)
            end = min(peak_tick + 50, 550)

            x_ticks = numpy.arange(start, end)

            fig = plt.figure(figsize=(16,9))
            plt.plot(x_ticks, sim_pmts[i_pmt][start:end], label=f"Generated PMT {i_pmt} signal")
            plt.plot(x_ticks, real_pmts[i_pmt][start:end], label=f"Real PMT {i_pmt} signal")
            plt.legend()
            plt.grid(True)
            plt.xlabel("Time Tick [us]")
            plt.ylabel("Amplitude")
            plt.savefig(plot_dir / pathlib.Path(f"pmt_{i_pmt}.png"))
            plt.tight_layout()
            plt.close()

        return


    # def plot_sipms(self, plot_dir, sim_sipms, real_sipms):

    #     x_ticks = numpy.arange(550)
    #     plot_dir.mkdir(parents=True, exist_ok=True)

    #     # Find the index of the peak sipm location:
    #     max_value = tf.reduce_max(real_sipms)
    #     max_loc   = tf.where(real_sipms == max_value)[0]

    #     # Draw the response sliced across the three dimensions:
    #     # Slice on z:    

    #     # This plots over all z, around the highest value sipm:
    #     central_i_x = max_loc[0]
    #     central_i_y = max_loc[1]
    #     central_i_z = max_loc[2]

    #     for i_x in [central_i_x -1, central_i_x, central_i_x + 1]:
    #         if i_x < 0 or i_x >= 47: continue
    #         for i_y in [central_i_y -1, central_i_y, central_i_y + 1]:
    #             if i_y < 0 or i_y >= 47: continue

    #             fig = plt.figure(figsize=(16,9))
    #             plt.plot(x_ticks, sim_sipms[i_x][i_y], label=f"Generated SiPM [{i_x}, {i_y}] signal")
    #             plt.plot(x_ticks, real_sipms[i_x][i_y], label=f"Real SiPM [{i_x}, {i_y}] signal")
    #             plt.legend()
    #             plt.grid(True)
    #             plt.xlabel("Time Tick [us]")
    #             plt.ylabel("Amplitude")
    #             plt.savefig(plot_dir / pathlib.Path(f"sipm_{i_x}_{i_y}.png"))
    #             plt.tight_layout()
    #             plt.close()

    #     # This plots x and y for a fixed z:
    #     for i_z in [central_i_z -1, central_i_z, central_i_z + 1]:
    #         if i_z < 0 or i_z >= 550: continue

    #         # 
    #         fig = plt.figure()
    #         plt.imshow(sim_sipms[:,:,i_z])
    #         plt.tight_layout()
    #         plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_slice_{i_z}.png"))
    #         plt.close()

    #         fig = plt.figure()
    #         plt.imshow(real_sipms[:,:,i_z])
    #         plt.tight_layout()
    #         plt.savefig(plot_dir / pathlib.Path(f"real_sipm_slice_{i_z}.png"))
    #         plt.close()


    # def plot_compressed_sipms(self, plot_dir, sim_sipms, real_sipms, axis):
        
    #     plot_dir.mkdir(parents=True, exist_ok=True)
        
    #     # What is the axis label for this compression?
    #     if axis == 0:
    #         label = "x"
    #     elif axis == 1:
    #         label = "y"
    #     elif axis == 2:
    #         label = "z"
    #     else:
    #         raise Exception(f"Invalid axis {axis} provided to compression plots.")

    #     # Compress time ticks:
    #     sim_comp = tf.reduce_sum(sim_sipms, axis=axis)

    #     fig = plt.figure()
    #     plt.imshow(sim_comp)
    #     plt.tight_layout()
    #     plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_compress_{label}.png"))
    #     plt.close()

    #     # Compress time ticks:
    #     real_comp = tf.reduce_sum(real_sipms, axis=axis)

    #     fig = plt.figure()
    #     plt.imshow(real_comp)
    #     plt.tight_layout()
    #     plt.savefig(plot_dir / pathlib.Path(f"real_sipm_compress_{label}.png"))
    #     plt.close()

    def comparison_plots(self, plot_directory):



        # In this function, we take the monitoring data, run an inference step,
        # And make plots of real vs sim responses.


        parameters = self.get_params(self.opt_state)

        self.key, subkey = jax.random.split(self.key)
        # First, run the monitor data through the simulator:
        simulated_pmts = self.simulate_fn(self.monitor_data['energy_deposits'], parameters, subkey)

        # Save the raw data into a file:
        print(plot_directory)
        plot_directory.mkdir(parents=True, exist_ok=True)
        # numpy.savez_compressed(plot_directory / pathlib.Path(f"output_arrays.npz"),
        #     real_pmts  = self.monitor_data["S2Pmt"],
        #     gen_pmts   = simulated_pmts,
        #     real_sipms = self.monitor_data["S2Si"],
        #     # gen_sipms  = gen_s2_si,

        #     )


        # # Now, instead of computing loss, we generate plots:


        batch_index=2

        pmt_dir = plot_directory / pathlib.Path(f"pmts/")
        self.plot_pmts(pmt_dir, simulated_pmts[batch_index], self.monitor_data["S2Pmt"][batch_index])

        # sim_data_3d  = gen_s2_si[batch_index]
        # real_data_3d = self.monitor_data["S2Si"][batch_index]


        # sipm_dir = plot_directory / pathlib.Path(f"sipms/")
        # self.plot_sipms(sipm_dir, sim_data_3d, real_data_3d)


        # # Take 2D compression views:
        # self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=0)
        # self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=1)
        # self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=2)




    def train_iteration(self, batch, i):

        # Max Lox:

        i_max = batch["S2Pmt"][0][0].argmax()

        metrics = {}

        parameters = self.get_params(self.opt_state)

        self.key, subkey = jax.random.split(self.key)

        (loss, waveforms), gradients = self.gradient_fn(batch, parameters, subkey)

        self.opt_state = self.opt_update(i, gradients, self.opt_state)

        print(batch["S2Pmt"][0][0][i_max - 5: i_max + 5])
        print(waveforms[0][0][i_max - 5: i_max + 5])


        logger.info(waveforms.max())
        logger.info(parameters["lifetime"])
        logger.info(parameters["diffusion"])
        logger.info(parameters["pmt_dynamic_range"])
        logger.info(parameters["waveform_sigma"])


        # self.parameters = apply_gradients(self.parameters, gradients, learning_rate=0.01)


        # gen_s2_pmt, gen_s2_si = self.simulate_fn(simulator(batch['energy_deposits']))
        # print(generated_s2_image.shape)



        # # s2_pmt_loss = tf.reduce_sum(tf.pow(batch["S2Pmt"] - gen_s2_pmt, 2.), axis=(1,2))
        # # s2_si_loss = tf.reduce_sum(tf.pow(batch["S2Si"] - gen_s2_si, 2.), axis=(1,2,3))
        # #
        # # s2_pmt_loss = tf.reduce_mean(s2_pmt_loss)
        # # s2_si_loss = tf.reduce_mean(s2_si_loss)

        # s2_pmt_loss = self.sipm_loss(batch["S2Pmt"],  gen_s2_pmt)
        # # s2_si_loss  = self.sipm_loss(batch["S2Si"],  gen_s2_si)
        # loss = s2_pmt_loss
        # # loss = s2_si_loss
        # # loss = s2_si_loss + s2_pmt_loss

        # reg = simulator.regularization()
        # # loss += reg

        # metrics['loss/s2_pmt_loss'] = s2_pmt_loss
        # # metrics['loss/s2_si_loss'] = s2_si_loss
        # metrics['loss/regularization'] = reg

        # # Combine losses

        metrics['loss/loss'] = loss


        # if MPI_AVAILABLE:
        #     tape = hvd.DistributedGradientTape(tape)


        # grads = tape.gradient(loss, simulator.trainable_variables)

        # self.optimizer.apply_gradients(zip(grads, simulator.trainable_variables))

        return metrics
