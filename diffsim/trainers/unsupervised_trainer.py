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
import flax.linen as nn

import pathlib
from matplotlib import pyplot as plt


def build_optimizer(config, params):

    opt_func_name = config.mode.optimizer.name

    optimizer = getattr(optax, opt_func_name)(
        config.mode.learning_rate,
        weight_decay = config.mode.weight_decay
    )

    # optimizer = optax.MultiSteps(optimizer, every_k_schedule=4)

    opt_state = optimizer.init(params)


    return optimizer, opt_state



# def close_over_training_step(config, MPI_AVAILABLE, sim_func, optimizer):
def close_over_training_step(config, MPI_AVAILABLE):

    # @jit
    def train_one_step(generator_state, discriminator_state, batch, rng_seeds):

        # This is like a GAN-style loop, except  the generator 
        # is physics-based.

        # The optimization proceedure looks like this:
        # - Take the energy depositions, and generate signals
        # - Take the generated signals, and discriminate on them
        # - Take the real signals, and discriminate on them
        # - Compute the discriminator loss for optimizing the discriminator
        # - Compute the discriminator loss for optimizing the generator
        # - Compute the gradients for whichever will be optimized this step (currently both)
        # - Apply the updates and return

        # In this, we say that 1.0 is "REAL" and 0.0 is "FAKE" 
        # in the output of the discriminator

        def g_loss_fn(g_params):

            # Generate signals:
            simulated_waveforms = generator_state.apply_fn(
                    g_params,
                    batch['e_deps'],
                    rngs=rng_seeds
                )
            # Discriminate against the generated signals:
            discriminated_gen = discriminator_state.apply_fn(
                discriminator_state.params,
                simulated_waveforms
            )

            # We try to get the generator close to 1.0 from the discriminator:
            g_loss = numpy.mean(optax.sigmoid_binary_cross_entropy(
                logits = discriminated_gen,
                labels = numpy.ones_like(discriminated_gen), 
            ))
            # g_loss = - numpy.mean(numpy.log(nn.sigmoid(discriminated_gen)))

            metrics = {
                "acc/gen"  : numpy.mean(discriminated_gen > 0.5),
                "loss/gen" : g_loss,
                "Mean/g-r" : numpy.mean(discriminated_gen)
            }

            return g_loss, metrics


        def d_loss_fn(d_params):


            # Generate signals:
            simulated_waveforms = generator_state.apply_fn(
                    generator_state.params,
                    batch['e_deps'],
                    rngs=rng_seeds
                )
            
            # Discriminate on the real signals:
            discriminated_real = discriminator_state.apply_fn(
                d_params,
                {"S2Si"  : batch["S2Si"], 
                 "S2Pmt" : batch["S2Pmt"]
                }
            )
            
            # discriminate on the generated signals:
            discriminated_gen = discriminator_state.apply_fn(
                d_params,
                simulated_waveforms
            )

            # Smooth the labels:
            alpha = 0.9
            smooth_ones_labels = alpha     * numpy.ones_like(discriminated_gen)
            smooth_zero_labels = (1-alpha) * numpy.ones_like(discriminated_gen)


            # print(discriminated_real)
            # print(discriminated_gen)
            # We want the discriminator to be correct, but with smooth 
            # labels so it doesn't get too confident.

            # Discriminator is putting the real labels close to 1.0:
            d_loss_real = numpy.mean(optax.sigmoid_binary_cross_entropy(
                logits = discriminated_real,
                labels = smooth_ones_labels,
            ))

            # And the fake labels close to 0:
            d_loss_gen = numpy.mean(optax.sigmoid_binary_cross_entropy(
                logits = discriminated_gen,
                labels = smooth_zero_labels,
            ))



            metrics = {
                "acc/disc-real" : numpy.mean(discriminated_real > 0.5),
                "acc/disc-gen" : numpy.mean(discriminated_gen <= 0.5),
                "loss/disc-real" : d_loss_real,
                "loss/disc-gen" : d_loss_gen,
                "Mean/d-g": numpy.mean(discriminated_gen),
                "Mean/d-r": numpy.mean(discriminated_real),
            }
            # print(d_loss_real.shape)
            # print(d_loss_gen.shape)

            return d_loss_gen, metrics
            # return d_loss_real + d_loss_gen, metrics


        d_grad_fn = jax.value_and_grad(d_loss_fn, has_aux=True)
        g_grad_fn = jax.value_and_grad(g_loss_fn, has_aux=True)

        # First, do the discriminator:
        (d_loss, d_metrics), d_grads = d_grad_fn(discriminator_state.params)
        (g_loss, g_metrics), g_grads = g_grad_fn(generator_state.params)

        g_metrics.update(d_metrics)


        if MPI_AVAILABLE:
            # Intercept here and allreduce the dict if necessary:
            d_grads = allreduce_dict(d_grads)
            g_grads = allreduce_dict(g_grads)

        # Apply the gradients with the optimizer:
        discriminator_state = discriminator_state.apply_gradients(grads=d_grads)
        generator_state     = generator_state.apply_gradients(grads=g_grads)


        return generator_state, discriminator_state, d_loss+g_loss, g_metrics

    return train_one_step




