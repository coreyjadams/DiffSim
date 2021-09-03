import tensorflow_probability as tfp
import tensorflow as tf

# Krypton Events are at 41.5575 keV, or 0.0415575 MeV. 
# The number of electrons produced is a gaussian distribution, 
# mean value is energy / (22.4 eV), variance is n_electrons * 0.15

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
# TFP expects the `scale` parameter == sigma == variance 

# We're using the new gaussian where, if X is drawn from the Normal distribution, 
# Y = aX + b

# We solve the equations from the reference above:
# a*mu_1 + b = mu_2
# a**2 sigma_1**2 = sigma_2**2

# Here, mu_1 = 0, and mu_2 = n, therefore b = n
# sigma_1 = 1.0, so a = sqrt(nF)

# In the end, to shift the distribution we multiply by variance and add the mean.

n_samples = 50000

normal_distribution = tfp.distributions.Normal(0.0, 1.0)

def generate_energies(normal_distribution : tfp.distributions.Distribution, energies):
    '''
    energies is expected in MeV
    '''

    print("energies:", energies)
    # For each energy, compute n:
    n = energies* 1000.*1000. / 22.4

    print("n:", n)
    sigmas = tf.sqrt(n * 0.15)

    print("sigmas:", sigmas)
    # Generate a sample for each energy:
    draws = normal_distribution.sample(len(energies))

    print("draws:", draws)

    # Shift with aX + b:
    electrons = sigmas * draws + n
    print("electrons:", electrons)


    return electrons


# Let's plot the distributions, one with krypton energies using the energy shift, and another drawn directly.

krypton_electron_gen = tfp.distributions.Normal(
            loc = 1855.2455357143, # number of electrons at 41.5575 keV
            scale = tf.sqrt(278.2868303571) # sigma at 41.5575 keV
        )


true_kr_electrons = krypton_electron_gen.sample(n_samples)

print("true_kr_electrons: ", true_kr_electrons)

kr_energies = 0.0415575 * tf.ones(shape=(n_samples,))

shifted_ke = generate_energies(normal_distribution, kr_energies)

print(shifted_ke)

from matplotlib import pyplot as plt
import numpy



true_kr_vals, bin_edges = numpy.histogram(true_kr_electrons)
bins_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
bins_widths  =     (bin_edges[1:] - bin_edges[:-1])

print(true_kr_vals)

shifted_kr_vals, _ = numpy.histogram(shifted_ke, bin_edges)
print(shifted_kr_vals)


plt.bar(bins_centers, true_kr_vals, width=bins_widths, label="True")
plt.bar(bins_centers, shifted_kr_vals, width=bins_widths, label="Shifted")
plt.legend()
plt.show()
