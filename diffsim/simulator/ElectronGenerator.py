import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial

class ElectronGenerator(nn.Module):
    """
    Class to turn energy depositions, at locations x/y/z/E, into
    (undiffused) electrons at the same location.

    Contains a parameter for the maximum number of electrons to generate.

    """


    p1: float
    p2: float
    n_max: int

    @partial(vmap, in_axes=[None, 0, 0])
    def energy_to_electrons(self, energy, normal_draw):
        """
        This function takes in an energy deposition and returns the number of electrons generated
        """

        # For each energy, compute n:
        n      = energy * 1000.*1000. / self.p1
        sigmas = numpy.sqrt( n * self.p2)
    
        # print(energy.shape)
        # print(sigmas.shape)
        # print(normal_draw.shape)

        # Generate a sample for each energy:
        n_electrons = (sigmas*normal_draw + n).astype(numpy.int32)


        return n_electrons


    @partial(vmap, in_axes=[None, 0])
    def broadcast_electron(self, position):
        """
        Broadcast the position of a single electron n_electrons
        times into a shape of n_max.
        """
        shape = (self.n_max, 3) # x/y/z per electron
        return numpy.broadcast_to(position, shape)

    @nn.compact
    def __call__(self, energies_and_positions):
        """
        Inputs is a sequence of energy depositions at x/y/z locations
        We turn it into a sequence of electrons at x/y/z locations
        """

        # First, split the energy and positions apart:
        positions = energies_and_positions[:,0:3]
        energies  = energies_and_positions[:,-1]

        rng = self.make_rng("electron")
        normal_draws = random.normal(rng, shape=energies.shape)

        # Get the number of electrons per position:
        n_electrons = self.energy_to_electrons(energies, normal_draws)

        # Turn the number of electrons into the right shape:

        broadcasted_electrons = self.broadcast_electron(positions)

        return broadcasted_electrons, n_electrons




def init_electron_generator(generator_params):

    EG = ElectronGenerator(
        p1 = generator_params.p1,
        p2 = generator_params.p2,
        n_max = generator_params.n_max,
    )


    return EG, ["electron",]