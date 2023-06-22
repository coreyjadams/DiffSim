import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial

class Lifetime(nn.Module):
    """
    Class to turn energy depositions, at locations x/y/z/E, into
    (undiffused) electrons at the same location.

    Contains a parameter for the maximum number of electrons to generate.

    """

    @partial(vmap, in_axes=[None, None,0 ])
    def generate_valid_mask(self, n_max, n_valid):
        '''
        Each cluster of electrons, of shape (n_max, 3), has n_valid electrons
        The rest should get set to 0.0.

        This is generating the mask only, though: not applying it.
        '''

        # This technique is "stupid" but it jit's and vmap's, so it's not THAT stupid

        # Generate a sequence of all the possible electrons by index
        arange = numpy.arange(n_max)
        # Select the ones less than the valid poitn:
        mask = arange < n_valid
        # Convert bool to float:
        mask = mask.astype("float32")

        return numpy.reshape(mask, (-1,1))


    @nn.compact
    def __call__(self, diffused_electrons, n_valid):
        """
        Pull off the energies, diffuse, stick back together, and return
        """

        n_max = diffused_electrons.shape[1]

        valid_mask = self.generate_valid_mask(n_max, n_valid)

        # We also draw a random number and use it to give each electron
        # a survival probability.  All the electrons have their probability
        # applied, so it's not stochastic but works on the ensemble only.

        lifetime_v = self.variable(
                "lifetime", "lifetime",
                lambda s : 5000.0*numpy.ones(s, dtype=diffused_electrons.dtype),
                (1,), # shape is scalar
            )
        lifetime = lifetime_v.value

        # Pull out the z value:
        z = diffused_electrons[:,:,2]

        probability = numpy.exp(-z / lifetime).reshape(valid_mask.shape)

        return valid_mask * probability




def init_lifetime(lifetime_params=None):

    lifetime = Lifetime()

    return lifetime, ["lifetime",]
