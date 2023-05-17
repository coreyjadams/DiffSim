import time


import jax
import jax.numpy as numpy
import jax.tree_util as tree_util
from jax import jit

from ..simulator.utils import batch_update_rng_keys

import logging
from logging import handlers

logger = logging.getLogger()

import optax

import pathlib
from matplotlib import pyplot as plt


def close_over_training_step(function_registry, config, MPI_AVAILABLE):

    @jit
    def compute_loss(simulated_signals, real_signals):
        # First, we compute the difference squared in the two:
        difference = (simulated_signals - real_signals)**2
        # Because this data is so sparse, we multiply the squared difference
        # by the input data to push the loss up in important places and
        # down in unimportant places.  But, don't want the loss to be zero
        # where the wavefunction is zero, so put a floor:
        loss = difference
        # loss = difference * (real_waveforms + 1e-4)

        # And, return the loss as a scalar:
        return loss.mean()


    @jit
    def forward(params, batch, rng_seeds):
        # print("type(params): ", type(params))
        # print("type(batch): ", type(batch))
        # # print("type(self.simulator_fn): ", type(simulator_fn))
        # print("batch['energy_deposits'].shape: ", batch['energy_deposits'].shape)
        # print("params.keys(): ", params.keys())
        # print("rng_seeds: ", rng_seeds)
        simulated_waveforms = function_registry["simulate"](
                params, 
                batch['energy_deposits'], 
                rngs=rng_seeds
            )
        # print(simulated_waveforms["pmt_s2"].shape)
        # print(simulated_waveforms["sipm_s2"].shape)
        loss_pmt = compute_loss(simulated_waveforms["pmt_s2"], batch['S2Pmt'])
        loss_sipm = compute_loss(simulated_waveforms["sipm_s2"], batch['S2Si'])
        loss = loss_pmt + loss_sipm
        return loss

    # @jit
    def train_one_step(sim_params, opt_state, batch, rng_seeds):
        print("forward")
        loss = forward(sim_params, batch, rng_seeds)
        print("grads")

        loss_value, grads = jax.value_and_grad(forward, argnums=0)(sim_params, batch, rng_seeds)
        print("updates")

        updates, opt_state = function_registry["optimizer"].update(grads, opt_state, sim_params)
        print("apply")

        sim_params = optax.apply_updates(sim_params, updates)

        return sim_params, opt_state, loss_value

    return train_one_step



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
            simulated_pmts, simulated_sipms = simulate_fn(batch['energy_deposits'], _parameters, key)
            loss_pmt = compute_loss(simulated_pmts, batch['S2Pmt'])
            loss_sipm = compute_loss(simulated_sipms, batch['S2Si'])
            loss = loss_pmt + loss_sipm
            # print(loss)
            return loss
            # return loss, ((loss_pmt, loss_sipm), (simulated_pmts, simulated_sipms))

        # self.gradient_fn = jit(jax.value_and_grad(forward_pass, argnums=1, has_aux=True))
        self.gradient_fn = jit(jax.value_and_grad(forward_pass, argnums=1, has_aux=False))





        opt_init, opt_update, get_params = jax_opt.adam(1e-3)
        # opt_init, opt_update, get_params = jax_opt.rmsprop(1e-3)
        # opt_init, opt_update, get_params = jax_opt.sgd(1e-3)

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


    def plot_sipms(self, plot_dir, sim_sipms, real_sipms):

        x_ticks = numpy.arange(550)
        plot_dir.mkdir(parents=True, exist_ok=True)


        # Find the index of the peak sipm location:
        max_value = numpy.max(real_sipms)
        max_x, max_y, max_z = numpy.unravel_index(numpy.argmax(real_sipms), real_sipms.shape)

        # This plots over all z, around the highest value sipm:
        for i_x in [max_x -1, max_x, max_x + 1]:
            if i_x < 0 or i_x >= 47: continue
            for i_y in [max_y -1, max_y, max_y + 1]:
                if i_y < 0 or i_y >= 47: continue

                print(sim_sipms[i_x][i_y][max_z-5:max_z+5])
                print(real_sipms[i_x][i_y][max_z-5:max_z+5])

                fig = plt.figure(figsize=(16,9))
                plt.plot(x_ticks, sim_sipms[i_x][i_y], label=f"Generated SiPM [{i_x}, {i_y}] signal")
                plt.plot(x_ticks, real_sipms[i_x][i_y], label=f"Real SiPM [{i_x}, {i_y}] signal")
                plt.legend()
                plt.grid(True)
                plt.xlabel("Time Tick [us]")
                plt.ylabel("Amplitude")
                plt.savefig(plot_dir / pathlib.Path(f"sipm_{i_x}_{i_y}.png"))
                plt.tight_layout()
                plt.close()

        # This plots x and y for a fixed z:
        for i_z in [max_z -1, max_z, max_z + 1]:
            if i_z < 0 or i_z >= 550: continue

            #
            fig = plt.figure()
            plt.imshow(sim_sipms[:,:,i_z])
            plt.tight_layout()
            plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_slice_{i_z}.png"))
            plt.close()

            fig = plt.figure()
            plt.imshow(real_sipms[:,:,i_z])
            plt.tight_layout()
            plt.savefig(plot_dir / pathlib.Path(f"real_sipm_slice_{i_z}.png"))
            plt.close()


    def plot_compressed_sipms(self, plot_dir, sim_sipms, real_sipms, axis):

        plot_dir.mkdir(parents=True, exist_ok=True)

        # What is the axis label for this compression?
        if axis == 0:
            label = "x"
        elif axis == 1:
            label = "y"
        elif axis == 2:
            label = "z"
        else:
            raise Exception(f"Invalid axis {axis} provided to compression plots.")

        # Compress time ticks:
        sim_comp = numpy.sum(sim_sipms, axis=axis)

        fig = plt.figure()
        plt.imshow(sim_comp)
        plt.tight_layout()
        plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_compress_{label}.png"))
        plt.close()

        # Compress time ticks:
        real_comp = numpy.sum(real_sipms, axis=axis)

        fig = plt.figure()
        plt.imshow(real_comp)
        plt.tight_layout()
        plt.savefig(plot_dir / pathlib.Path(f"real_sipm_compress_{label}.png"))
        plt.close()

    def parameters(self):

        p = self.get_params(self.opt_state)
        # Deliberately slice things up here:

        parameters = {}
        parameters["diffusion/x"] = p["diffusion"][0]
        parameters["diffusion/y"] = p["diffusion"][1]
        parameters["diffusion/z"] = p["diffusion"][2]
        parameters["lifetime"] = p["lifetime"]
        parameters["el_spread"] = p["el_spread"]
        # parameters["sipm_scale_mean"] = p["sipm_dynamic_range"].mean()
        # parameters["sipm_scale_std"] = p["sipm_dynamic_range"].std()
        # parameters["el_amplification"] = p["el_amplification"]
        return parameters

    def comparison_plots(self, plot_directory):



        # In this function, we take the monitoring data, run an inference step,
        # And make plots of real vs sim responses.


        parameters = self.get_params(self.opt_state)

        self.key, subkey = jax.random.split(self.key)
        # First, run the monitor data through the simulator:
        simulated_pmts, simulated_sipms = self.simulate_fn(self.monitor_data['energy_deposits'], parameters, subkey)

        # Save the raw data into a file:
        plot_directory.mkdir(parents=True, exist_ok=True)
        # numpy.savez_compressed(plot_directory / pathlib.Path(f"output_arrays.npz"),
        #     real_pmts  = self.monitor_data["S2Pmt"],
        #     gen_pmts   = simulated_pmts,
        #     real_sipms = self.monitor_data["S2Si"],
        #     # gen_sipms  = gen_s2_si,

        #     )


        # # Now, instead of computing loss, we generate plots:


        batch_index=0

        pmt_dir = plot_directory / pathlib.Path(f"pmts/")
        self.plot_pmts(pmt_dir, simulated_pmts[batch_index], self.monitor_data["S2Pmt"][batch_index])

        sim_data_3d  = simulated_sipms[batch_index]
        real_data_3d = self.monitor_data["S2Si"][batch_index]


        sipm_dir = plot_directory / pathlib.Path(f"sipms/")
        self.plot_sipms(sipm_dir, sim_data_3d, real_data_3d)


        # Take 2D compression views:
        self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=0)
        self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=1)
        self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=2)




    def train_iteration(self, batch, i, parameters):

        metrics = {}

        # parameters = self.get_params(self.opt_state)

        self.key, subkey = jax.random.split(self.key)


        # (loss, (loss_pmts, loss_sipms), (pmts, sipms) ), gradients = self.gradient_fn(batch, parameters, subkey)
        loss, gradients = self.gradient_fn(batch, parameters, subkey)

        parameters = jax.tree_util.tree_map(
            lambda x, y: x - 1e-5*y, parameters, gradients)


        # print(parameters["el_spread"], gradients["el_spread"])
        # self.opt_state = self.opt_update(i, gradients, self.opt_state)


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

        metrics.update(self.parameters())
        # if MPI_AVAILABLE:
        #     tape = hvd.DistributedGradientTape(tape)


        # grads = tape.gradient(loss, simulator.trainable_variables)

        # self.optimizer.apply_gradients(zip(grads, simulator.trainable_variables))

        return parameters, metrics
