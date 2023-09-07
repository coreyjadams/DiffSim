import jax.numpy as numpy
import jax.random as random

from jax import tree_util, vmap
from functools import reduce

from . discriminator import init_discriminator_model


def init_discriminator(init_key, config, example_data):


    import flax

    from jax import vmap, jit

    disc = init_discriminator_model()

    d_params = disc.init(init_key, 
        {
            "S2Si": example_data["S2Si"][0],
            "S2Pmt": example_data["S2Pmt"][0],
        }
    )

    disc_fn = jit(vmap(disc.apply, in_axes= (None, 0,)))


    # Initialize the simulator object, which has a number of 
    # hooks for jax RNGs and we need to see them too

    # Initialize the parameters:

    return disc_fn, d_params