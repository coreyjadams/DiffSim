import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial

class Diffusion(nn.Module):
    """
    Class to turn energy depositions, at locations x/y/z/E, into
    (undiffused) electrons at the same location.

    Contains a parameter for the maximum number of electrons to generate.

    """

    @partial(vmap, in_axes=[None, 0,0])
    def diffuse_electrons(self, electrons, kicks):
        '''
        Apply diffusion to a single energy deposition (shape of [N, 3])
        '''

        # Diffusion is proportional to the sqrt of Z:
        z = electrons[:,3]
        # Need to reshape it to have the x/y/z dimension for broadcasting
        scale = numpy.sqrt(z).reshape((-1,1))

        # Get the diffusion variable:
        is_initialized = self.has_variable("diffusion", "diffusion")
        diffusion_v = self.variable(
                "diffusion", "diffusion",
                lambda s : 0.1*numpy.ones(s, dtype=electrons.dtype),
                electrons[0].shape
            )
        # This actually fetches the value:
        diffusion = diffusion_v.value
        # Apply it as a correction that is proportional to sqrt(z) but has a normal component too
        return electrons + diffusion * kicks * scale

    @nn.compact
    def __call__(self, electrons):
        """
        Pull off the energies, diffuse, stick back together, and return
        """

        # Generate a unit normal distribution of kicks:
        rng = self.make_rng("diffusion")
        normal_draws = random.normal(rng, shape=electrons.shape)

        # The kicks get scaled by sqrt(z) and diffusion strength here:
        kicked =  self.diffuse_electrons(electrons, normal_draws)

        return kicked



# def init_electron_generator(generator_params):

#     EG = Diffusion(
#         p1 = generator_params.p1,
#         p2 = generator_params.p2,
#         n_max = generator_params.n_out,
#     )

#     return EG