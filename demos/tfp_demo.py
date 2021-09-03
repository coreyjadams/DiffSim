import tensorflow as tf
import tensorflow_probability as tfp


# In this demo, let's regress on the parameters of a random distribution to best 
# fit a know distribution.

target_distribution = tfp.distributions.Normal(loc=1.0, scale=0.5)




learnable_distribution = tfp.distributions.Normal(
    loc = tf.Variable(0.0),
    scale = tf.Variable(1.0))

def energy_distance(
    sample1 : tfp.distributions.Distribution, 
    sample2 : tfp.distributions.Distribution,
    n_samples = 100):
    ''' Compute the energy distance of two DISTRIBUTIONS.

    '''

    def sample_distance(s1, s2):


        n_s1 = s1.shape[0]
        n_s2 = s2.shape[0]

        # We need pairwise distances, which we get with repeats:
        r1 = tf.repeat(s1, (n_s2,))
        r1 = tf.reshape(r1, (n_s1, n_s2))

        # Compute the pairwise distance, sum, and divide by total number of samples:
        return tf.reduce_sum(tf.abs(r1 - s2)) / (n_s1 * n_s2)

    A = sample_distance(sample1.sample(n_samples), sample2.sample(n_samples))
    B = sample_distance(sample1.sample(n_samples), sample1.sample(n_samples))
    C = sample_distance(sample2.sample(n_samples), sample2.sample(n_samples))

    return 2*A - B - C

def energy_distance_fast(
    sample1 : tfp.distributions.Distribution, 
    sample2 : tfp.distributions.Distribution,
    n_samples = 100):

    ''' This is an approximate to energy distance that forces both distributions
    to have the same sample size.

    This assumption lets us avoid broadcasting and operate much faster
    '''

    def sample_distance(s1, s2):

        # We need pairwise distances, which we get with repeats:
        
        # Compute the pairwise distance, sum, and divide by total number of samples:
        return tf.reduce_mean(tf.abs(s1 - s2))

    A = sample_distance(sample1.sample(n_samples), sample2.sample(n_samples))
    B = sample_distance(sample1.sample(n_samples), sample1.sample(n_samples))
    C = sample_distance(sample2.sample(n_samples), sample2.sample(n_samples))

    return 2*A - B - C



n_samples = 5000

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

# Train the distributions to match by minizing energy distance:
for i in range(1000):

    with tf.GradientTape() as tape:
        dist = energy_distance_fast(target_distribution, learnable_distribution, n_samples)

    grads = tape.gradient(dist, learnable_distribution.trainable_variables )

    optimizer.apply_gradients(zip(grads, learnable_distribution.trainable_variables))

    print(dist)

print(energy_distance(target_distribution, learnable_distribution))

print(learnable_distribution.trainable_variables)

