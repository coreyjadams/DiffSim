import tensorflow as tf
import tensorflow_probability as tfp


# We can not create a new gaussian efficiently at every point.
# Instead, for each deposition we need, we sample a normal distribution
# and then shift every point by the needed mean and variance.

# Reference:
# https://math.stackexchange.com/questions/2894681/how-to-transform-shift-the-mean-and-standard-deviation-of-a-normal-distribution

# We sample from a normal distribution with a mean of 0 and a variance of 1.
# For any particular energy, E, we need to have samples from a distribution
# where the mean value is n, and the variance is sqrt(nF).  here, n = E / 22.4 eV, 
# and F = 0.15

# variance == sigma**2, so here sigma = nF.
# TFP expects the `scale` parameter == sigma == sqrt(nF)

# We're using the new gaussian where, if X is drawn from the Normal distribution, 
# Y = aX + b

# We solve the equations from the reference above:
# a*mu_1 + b = mu_2
# a**2 sigma_1**2 = sigma_2**2

# Here, mu_1 = 0, and mu_2 = n, therefore b = n
# sigma_1 = 1.0, so a = sqrt(nF)

# In the end, to shift the distribution we multiply by variance and add the mean.


class ElectronGenerator:

    def __init__(self):


        self.normal_distribution = tfp.distributions.Normal(
            loc = 0.0, 
            scale = 1.0
        )

    def generate_electrons(self,
        energy_depositions : tf.Tensor):
        '''
        energies is expected in MeV
        expected to be [N, 4] tensor with [x/y/z]
        '''

        electron_positions = energy_depositions[:, 0:3]
        energies = energy_depositions[:, -1]
        

        # For each energy, compute n:
        n = energies* 1000.*1000. / 22.4

        sigmas = tf.sqrt(n * 0.15)

        # Generate a sample for each energy:
        draws = self.normal_distribution.sample(len(energies))
        draws = tf.reshape(draws, (len(energies), 1))

        # Shift with aX + b:
        electrons = sigmas * draws + n


        electrons_and_postions = tf.concat((electron_positions, electrons), axis=-1)


        return electrons_and_postions






