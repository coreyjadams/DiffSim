import time


import jax
import jax.numpy as numpy
import jax.tree_util as tree_util
from jax import jit, vmap

from ..simulator.utils import batch_update_rng_keys

from diffsim.utils import allreduce_dict

import logging
from logging import handlers

logger = logging.getLogger()

import optax

import pathlib
from matplotlib import pyplot as plt


# def close_over_training_step(config, MPI_AVAILABLE, sim_func, optimizer):
def close_over_training_step(config, MPI_AVAILABLE):

    @jit
    def compute_log_loss(simulated_signals, real_signals):
        '''
        Compute loss over a single event.
        Gets vmap'd over a batch.
        '''
        power = config.mode.loss_power

        # First, we compute the difference in the two signals w/ abs value:
        difference = numpy.abs(simulated_signals - real_signals)**power
        print(difference)
        # The power is sort of like a focal term.

        # Next compute the log of this difference, with a baseline term to prevent
        # it from going negative:
        # Why compute the log?  Because the loss is SO HIGH at the start
        # Adding 1.0 contributes nothing to the loss when the signals are equal.
        loss = numpy.log(difference + 1.)

        # Optionally, may increase a with a focal term:

        # # Because this data is so sparse, we multiply the difference
        # # by the input data to push the loss up in important places and
        # # down in unimportant places.  But, don't want the loss to be zero
        # # where the wavefunction is zero, so put a floor:
        # mask = real_signals > 0.1
        # # Cast it to floating point:
        # mask = mask.astype("float32")
        # # Here's the floor:
        # weight = 1e-4*numpy.ones(difference.shape)
        # # Amplify the non-zero regions
        # weight = weight + mask #(so the loss is either 1e-4 or 1.0001)
        #
        # difference = difference * weight


        # Take the sum of the difference over the last axis, the waveform:
        loss = numpy.sum(loss, axis=-1)
        # loss = numpy.sum(loss, axis=-1) / numpy.sum(weight)

        # Take the mean of the loss over the sensor arrays:
        loss = loss.mean()

        # We weigh the loss by the integral of the signal too:

        # And, return the loss as a scalar:
        return loss

    @jit
    def compute_residual(simulated_signals, real_signals):
        # Here it is measured over each waveform:
        residual = (simulated_signals - real_signals)**2
        # Sum over the last axis, time times, and sqrt:
        residual = numpy.sqrt(residual.sum(axis=-1))

        # Take the mean of this over all waveforms:
        return numpy.mean(residual)

    @jit
    def train_one_step(state, batch, rng_seeds):


        def loss_fn(params):
            simulated_waveforms = state.apply_fn(
                    params,
                    batch['e_deps'],
                    rngs=rng_seeds
                )
            

            # Compute the loss, mean over the batch:
            loss = {
                key : vmap(compute_log_loss, in_axes=(0,0))(
                    simulated_waveforms[key],
                    batch[key]).mean()
                for key in ["S2Si", "S2Pmt"]
            }
            # loss = {}
            # loss["S2Pmt"] = optax.huber_loss(simulated_waveforms["S2Pmt"], batch["S2Pmt"])
            # loss["S2Si"]  = optax.l2_loss(simulated_waveforms["S2Si"], batch["S2Si"])


            # loss["S2Pmt"] = numpy.sum(loss["S2Pmt"], axis=-1)
            # loss["S2Pmt"] = numpy.mean(loss["S2Pmt"])

            # loss["S2Si"] = numpy.sum(loss["S2Si"], axis=-1)
            # loss["S2Si"] = numpy.mean(loss["S2Si"])

            # print(loss["S2Pmt"].shape)
            # print(loss["S2Si"].shape)

            # Compute the residual which is an unweight, unnormalized comparison:
            metrics = {
                "residual/" + key : vmap(compute_residual, in_axes=(0,0))(simulated_waveforms[key], batch[key]).mean()
                for key in ["S2Si", "S2Pmt"]
            }

            metrics.update(
                { "loss/" + key : loss[key] for key in loss.keys() }
            )

            # loss =  loss["S2Pmt"]
            # loss =  loss["S2Si"]
            loss = config.mode.s2pmt_scaling * loss["S2Pmt"] + config.mode.s2si_scaling * loss["S2Si"]

            return loss, metrics

        # print(state.apply_fn)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (loss, metrics), grads = grad_fn(state.params)


        if MPI_AVAILABLE:
            # Intercept here and allreduce the dict if necessary:
            grads = allreduce_dict(grads)

        # Apply the gradients with the optimizer:
        state = state.apply_gradients(grads=grads)

        # (loss_value, metrics), grads = jax.value_and_grad(forward, argnums=0, has_aux=True)(
        #     state, batch, rng_seeds)



        return state, loss, metrics

    return train_one_step




