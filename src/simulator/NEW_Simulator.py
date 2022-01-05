import tensorflow as tf
import tensorflow_probability as tfp


class NEXT_Simulator(tf.keras.Model):

    def __init__(self):
        tf.keras.Model.__init__(self)
        self.n_pmts  = 12
        self.n_ticks = 550

        self.diffusion = tfp.distributions.MultivariateNormalDiag(
            loc        = [
                0.0,
                0.0,
                0.0,
            ],
            scale_diag = [
                1.0,
                1.0,
                1.0,
            ]
        )

        # Trainable Parameters for diffusion:
        self.diffusion_scale = tf.Variable([1.0, 1.0, 0.55])

        self.electron_normal_distribution = tfp.distributions.Normal(
            loc = 0.0,
            scale = 1.0
        )


        # Here is the pmt network:
        self.s2pmt_layer1 = tf.keras.layers.Dense(units=28, activation="sigmoid")
        self.s2pmt_layer2 = tf.keras.layers.Dense(units=12, activation="sigmoid")

        # Here is the sipm network:
        # Need to end up with 47x47 = 2209 sipms.
        self.s2si_layer1 = tf.keras.layers.Dense(units=64,   activation="sigmoid")
        self.s2si_layer2 = tf.keras.layers.Dense(units=128,  activation="sigmoid")
        self.s2si_layer3 = tf.keras.layers.Dense(units=256,  activation="sigmoid")
        self.s2si_layer4 = tf.keras.layers.Dense(units=47*47, activation="sigmoid")


        # PMT Response scale:
        self.pmt_response_scale = tf.Variable(tf.ones(shape=[12]))

        # Sipm Response scale:
        self.si_response_scale = tf.Variable(tf.ones(shape=[47*47]))


        # Lifetime variables:
        self.lifetime = tf.Variable(12000.)
        self.lifetime_sharpness = tf.constant(0.001)
        self.uniform_sampler = tfp.distributions.Uniform()

        # Handling the z-differentiation with a gaussian response along Z for each sensor.
        self.bin_sigma = tf.constant(0.1)
        self.gaussian_norm = tf.constant(1./(self.bin_sigma * 2.5066282746) )

    def generate_summary_dict(self):
        # Add relevant objects to a dictionary for a summary:

        metrics = {}

        metrics['diffusion/x'] = self.diffusion_scale[0]
        metrics['diffusion/y'] = self.diffusion_scale[1]
        metrics['diffusion/z'] = self.diffusion_scale[2]

        metrics['lifetime']    = self.lifetime.numpy()

        return metrics
    # @profile
    # @tf.function
    def s2_call(self, inputs, diffusion_weight, training=True):

        responses = [ self.s2_subcall(i, d) for i, d in zip(inputs, diffusion_weight) ]

        pmt_responses, sipm_responses = zip(*responses)

        pmt_response  = tf.stack(pmt_responses)
        sipm_response = tf.stack(sipm_responses)

        # print("Final pmt_response.shape: ", pmt_response.shape)
        # print("Final sipm_response.shape: ", sipm_response.shape)

        return pmt_response, sipm_response


    # @tf.function
    # @profile
    def s2_subcall(self, electrons, weight):
        # Pull out z_ticks:
        z_ticks = electrons[:,2]
        starts = tf.zeros(shape=z_ticks.shape) + 0.5
        stops  = tf.ones(shape=z_ticks.shape) * (self.n_ticks - 1) + 0.5
        # Get the x/y locations
        xy_electrons = electrons[:,0:2]

        # Reshape the ticks:
        z_ticks = tf.reshape(z_ticks, z_ticks.shape + (1,))
        # Stack the tick ranges, one per tick:
        # print(z_ticks.shape)
        # exp_input = tf.stack( [tf.range(self.n_ticks,dtype=tf.dtypes.float32) for _z in z_ticks ])
        exp_input = tf.linspace(start=starts, stop=stops, num=550,axis=-1)


        # print(exp_input.shape)
        # Apply the exponential, transpose, and make sparse:

        exp_values = self.gaussian_norm * tf.exp( -(exp_input - z_ticks)**2 / self.bin_sigma)
        #
        # print(exp_values.shape)
        # print(weight.shape)
        # print((tf.reshape(weight, weight.shape + (1,))*exp_values).shape)

        # Multiple by the weight of each electron:
        exp_values = tf.reshape(weight, weight.shape + (1,))*exp_values

        z_values_sparse = tf.sparse.from_dense(tf.transpose(exp_values))
        # print(z_values_sparse)

        # This runs the neural network to get the PMT and SiPM response:
        pmt_response  = self.s2pmt_call_network(xy_electrons)
        pmt_result  = tf.sparse.sparse_dense_matmul(z_values_sparse, pmt_response)
        pmt_result = tf.transpose(pmt_result)

        sipm_response = self.s2si_call_network(xy_electrons)
        sipm_result = tf.sparse.sparse_dense_matmul(z_values_sparse, sipm_response)
        sipm_result = tf.transpose(sipm_result)
        sipm_result = tf.reshape(sipm_result, (47, 47, self.n_ticks))



        return pmt_result, sipm_result

    # @profile
    def s2pmt_call_network(self,xy_electrons):
        x = self.s2pmt_layer1(xy_electrons)
        x = self.s2pmt_layer2(x)
        return x*tf.math.pow(self.pmt_response_scale, 2)

    def s2si_call_network(self,xy_electrons):
        x = self.s2si_layer1(xy_electrons)
        x = self.s2si_layer2(x)
        x = self.s2si_layer3(x)
        x = self.s2si_layer4(x)
        # x = tf.reshape(x, (-1, 47, 47))
        # print(x.shape)
        return x*tf.math.pow(self.si_response_scale, 2)


    # @profile
    def generate_electrons(self,
        energy_depositions : tf.Tensor):
        '''
        energies is expected in MeV
        expected to be [B, N, 4] tensor with [x/y/z/E]
        '''

        electron_positions = energy_depositions[:,:, 0:3]
        energies = energy_depositions[:,:,3]

        # For each energy, compute n:
        n = energies* 1000.*1000. / 22.4

        sigmas = tf.sqrt(n * 0.15)

        # Generate a sample for each energy:
        draws = self.electron_normal_distribution.sample(energies.shape)

        # Shift with aX + b:
        electrons = tf.cast(sigmas * draws + n, tf.dtypes.int32)

        return electrons, electron_positions


    # @profile
    def diffuse_electrons(self, n_electrons, positions):

        # Sample the diffusion of electrons once per electon, but ensure to multiply by sqrt(z)

        BATCH_SIZE = n_electrons.shape[0]
        MAX_DEPOSITIONS = n_electrons.shape[-1]

        z = positions[:,:,-1]



        sqrt_z = tf.math.sqrt(z)

        # We sample for the TOTAL number of electrons:
        total_samples = tf.reduce_sum(n_electrons)

        samples_per_batch = tf.reduce_sum(n_electrons, axis=-1)

        # Do the sampling, yields a 3D vector from the central location for each electron:
        samples = self.diffusion.sample(total_samples)

        # Split the sampled electrons into the piles per energy deposition
        displacements = tf.split(samples,tf.reshape(n_electrons, (-1,)), axis=0)

        # print(len(displacements))

        # At this point, each individual energy deposition has a vector of displacements, for each electron
        sqrt_z = tf.reshape(sqrt_z, (-1))
        for i, (scale, mean) in enumerate(zip(tf.reshape(sqrt_z, (-1,)), tf.reshape(positions,(-1,3) ) )):
            displacements[i] = tf.math.pow(self.diffusion_scale, 2)*displacements[i]*scale + mean
        # Now, we don't care about the individual depositions anymore, we have all of the electrons.

        # We recombine into per-batch events to help smooth this over:
        diffused_electrons = [ tf.concat(displacements[i*MAX_DEPOSITIONS:(i+1)*MAX_DEPOSITIONS], axis=0) for i in range(BATCH_SIZE)]

        # Apply the lifetime right here:
        # diffused_electrons = [ self.apply_lifetime(de) for de in diffused_electrons]

        return diffused_electrons

    # @tf.function
    def select_electrons(self, _input_electrons):
        z_position = _input_electrons[:,2]
        # We simply compute a weight, per electron, dependent on the lifetime

        # One random number per electron:
        randoms = self.uniform_sampler.sample(z_position.shape[0])

        threshold = tf.math.exp( - z_position / self.lifetime) - randoms
        threshold /= self.lifetime_sharpness
        # This maps each electron into roughly a zero or one value
        weight = tf.math.sigmoid(threshold)
        return weight


    # @profile
    def apply_lifetime(self, diffused_electrons):

        # @profile


        selected_electrons = [ self.select_electrons(de) for de in diffused_electrons]

        return selected_electrons

    # @profile
    def call(self, energy_depositions, training=True):
        n_electrons, positions = self.generate_electrons(energy_depositions)
        diffused_electrons = self.diffuse_electrons(n_electrons, positions)
        diffusion_weight = self.apply_lifetime(diffused_electrons)
        s2pmt, s2si = self.s2_call(diffused_electrons, diffusion_weight)
        return s2pmt, s2si
