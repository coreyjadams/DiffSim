import time


import jax
import jax.numpy as numpy
import jax.tree_util as tree_util
from jax import jit, vmap

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
        '''
        Compute loss over a single event.
        Gets vmap'd over a batch.
        '''
        power = config.mode.loss_power

        # First, we compute the difference squared in the two:
        difference = numpy.abs(simulated_signals - real_signals)**power

        # Because this data is so sparse, we multiply the difference
        # by the input data to push the loss up in important places and
        # down in unimportant places.  But, don't want the loss to be zero
        # where the wavefunction is zero, so put a floor:
        mask = real_signals > 0.1
        # Cast it to floating point:
        mask = mask.astype("float32")
        # Here's the floor:
        weight = 1e-4*numpy.ones(difference.shape)
        # Amplify the non-zero regions
        weight = weight + mask

        difference = difference * weight


        # Take the sum of the difference over the last axis, the waveform:
        difference = numpy.sum(difference, axis=-1)

        # Take the mean of the difference over the sensor arrays:
        difference = difference.mean()

        # We weigh the difference by the integral of the signal too:

        # And, return the loss as a scalar:
        return difference

    @jit
    def compute_residual(simulated_signals, real_signals):
        # Here it is measured over each waveform:
        residual = (simulated_signals - real_signals)**2
        # Sum over the last axis, time times, and sqrt:
        residual = numpy.sqrt(residual.sum(axis=-1))

        # Take the mean of this over all waveforms:
        return numpy.mean(residual)

    @jit
    def forward(params, batch, rng_seeds):
        simulated_waveforms = function_registry["simulate"](
                params, 
                batch['energy_deposits'], 
                rngs=rng_seeds
            )
        
        # Compute the loss, mean over the batch:
        loss = {
            key : vmap(compute_loss, in_axes=(0,0))(
                simulated_waveforms[key], 
                batch[key]).mean()
            for key in ["S2Si", "S2Pmt"]
        }


        residual = {
            "residual/" + key : vmap(compute_residual, in_axes=(0,0))(simulated_waveforms[key], batch[key]).mean()
            for key in ["S2Si", "S2Pmt"]
        }

        residual.update(
            { "loss/" + key : loss[key] for key in loss.keys() }
        )

        # loss = loss["S2Pmt"]
        loss = loss["S2Pmt"] + loss["S2Si"]

        return loss, residual

    @jit
    def train_one_step(sim_params, opt_state, batch, rng_seeds):

        (loss_value, metrics), grads = jax.value_and_grad(forward, argnums=0, has_aux=True)(
            sim_params, batch, rng_seeds)

        updates, opt_state = function_registry["optimizer"].update(grads, opt_state, sim_params)

        sim_params = optax.apply_updates(sim_params, updates)

        return sim_params, opt_state, loss_value, metrics

    return train_one_step




    # def parameters(self):

    #     p = self.get_params(self.opt_state)
    #     # Deliberately slice things up here:

    #     parameters = {}
    #     parameters["diffusion/x"] = p["diffusion"][0]
    #     parameters["diffusion/y"] = p["diffusion"][1]
    #     parameters["diffusion/z"] = p["diffusion"][2]
    #     parameters["lifetime"] = p["lifetime"]
    #     parameters["el_spread"] = p["el_spread"]
    #     # parameters["sipm_scale_mean"] = p["sipm_dynamic_range"].mean()
    #     # parameters["sipm_scale_std"] = p["sipm_dynamic_range"].std()
    #     # parameters["el_amplification"] = p["el_amplification"]
    #     return parameters
