import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial
from .MLP import MLP, init_mlp


class Diffusion(nn.Module):
    """
    Class to turn energy depositions, at locations x/y/z/E, into
    (undiffused) electrons at the same location.

    Contains a parameter for the maximum number of electrons to generate.

    """

    drift_velocity: MLP


    @partial(vmap, in_axes=[None, 0,0])
    def diffuse_electrons(self, electrons, kicks):
        '''
        Apply diffusion to a single electron (shape of [3])
        '''
        print(electrons.shape)

        # Diffusion is proportional to the sqrt of Z:
        z = electrons[2]
        # The absolute scale of diffusion depends on the Z location:
        scale = numpy.sqrt(z)

        # # Assuming the drift velocity might not be constant over the whole range:
        # drift_velocity_correction = self.drift_velocity(z.reshape(-1,1)).reshape(z.shape)

        # # Apply it as a "residual" type correction of max size ~25%
        # z = (1+ 0.05*drift_velocity_correction)*z

        # Need to reshape it to have the x/y/z dimension for broadcasting

        # Get the diffusion variable:
        is_initialized = self.has_variable("diffusion", "diffusion")
        diffusion_v = self.variable(
                "params", "diffusion",
                lambda s : .01*numpy.ones(s, dtype=electrons.dtype),
                electrons.shape
            )
        # This actually fetches the value:
        diffusion = diffusion_v.value 
        # Apply it as a correction that is proportional to sqrt(z) but has a normal component too
        diffusion =  diffusion * kicks * scale

        # Replace the z position with the scaled z by the drift V correction:
        # electrons = electrons.at[:,3].set(z)

        return electrons + diffusion

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



def init_diffusion(diffusion_params):

    mlp_config = diffusion_params.mlp_cfg
    mlp, _ = init_mlp(mlp_config, nn.sigmoid)

    diff = Diffusion(mlp)

    return diff, ["diffusion",]
