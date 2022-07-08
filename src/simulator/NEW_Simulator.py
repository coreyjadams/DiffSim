import tensorflow as tf
import tensorflow_probability as tfp

MAX_ELECTRONS = 2000
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
            scale_diag = tf.Variable([
                1.0,
                1.0,
                1.0,
            ])
        )

        # Trainable Parameters for diffusion:
        self.diffusion_scale = tf.constant([1.0, 1.0, 1.0])

        self.electron_normal_distribution = tfp.distributions.Normal(
            loc   = 0.0,
            scale = 1.0
        )

        # Need to create an array of all sipm locations.
        # Note that locations are in mm, and the center at 0,0
        # Create the 1D locations:
        sipms_1D = tf.range(-235., 235., 10.)
        n_sipms = sipms_1D.shape[0]
        # Map 1D locations to a tile for X, tile for Y:
        sipm_locations_x = tf.tile(sipms_1D, (n_sipms,))
        sipm_locations_y = tf.tile(sipms_1D, (n_sipms,))
        # Put them in the right shape, and transpose x:
        sipm_locations_x = tf.transpose(tf.reshape(sipm_locations_x, (n_sipms, n_sipms)))
        sipm_locations_y = tf.reshape(sipm_locations_y, (n_sipms, n_sipms))
        # Stack together"
        sipm_locations = tf.stack([sipm_locations_x, sipm_locations_y], -1)
        # Reshape since that's needed for the broadcast:
        self.sipm_locations = tf.reshape(sipm_locations, (1,) + sipm_locations.shape)

        # Here is the pmt network:
        self.pmt_network = tf.keras.Sequential([
                tf.keras.layers.Dense(units=28, activation="sigmoid"),
                tf.keras.layers.Dense(units=28, activation="sigmoid"),
                tf.keras.layers.Dense(units=28, activation="sigmoid"),
                tf.keras.layers.Dense(units=12, activation="sigmoid"),
            ])

        # Sipms are generally a gaussian-like response to incident charge.
        # We set a trainable sigma in x and y and then fine tune that.
        self.sipm_sigma = tf.Variable(1.)

        # We need a small network to model the EL amplification.
        # Takes an XY location and provides and overall normalization
        # to apply to all sipms (same per sipm)

        # Overall, adc per electron into the EL region for Sipms:
        self.sipm_response_normalization = tf.Variable(100.0)

        self.el_distance = tf.Variable(1.)

        #  Overall light production (normalization of PSF) is modulated by a network:
        self.s2si_layer1 = tf.keras.layers.Dense(units=28, activation="sigmoid")
        self.s2si_layer2 = tf.keras.layers.Dense(units=1, activation="tanh")
        # This network above outputs a value from -1 to 1.  
        # We multiply by 0.5, add it to 1.0 manually, and the output is from 0.5 to 1.5
        # And then it is multiplicative to the total response normalization above.  
        # So, this becomes a modulation up to a factor 



        # PMT Response scale:
        self.pmt_response_scale = tf.Variable(tf.ones(shape=[12]))

        # Sipm Response scale:
        # (sipm by sipm variations)
        self.si_response_scale = tf.Variable(tf.ones(shape=[47*47]))


        # Lifetime variables:
        self.lifetime = tf.Variable(3000.)


        # Handling the z-differentiation with a gaussian response along Z for each sensor.
        # In order to have a guassian response that is fairly normalized, this should be not
        # too small compared to the bin width.
        # 
        # N.B. : Fluctuations in the normalization occur if the gaussian varies quickly 
        # with respect to the sampling frequency. Practicially, with 1 us bins, 0.2 is a good range.
        # Higher sigma smears the signal to neighboring ticks more, while lower sigma
        # introduces small fluctuations based on sampling inefficiency
        # 
        # On average, 0.2 puts 75% of the response in the right bin, with 1 to 3% normalization wiggle.
        self.bin_sigma = tf.constant(0.2)
        # Norm is 1/(sigma * sqrt(2pi))
        self.gaussian_norm = tf.constant(1./(tf.sqrt(self.bin_sigma) * 2.5066282746) )

    def generate_summary_dict(self):
        # Add relevant objects to a dictionary for a summary:

        metrics = {}


        metrics['diffusion/x'] = self.diffusion_scale[0]
        metrics['diffusion/y'] = self.diffusion_scale[1]
        metrics['diffusion/z'] = self.diffusion_scale[2]

        metrics['sipm_psf']    = self.sipm_sigma
        metrics['sipm_amplitude'] = self.sipm_response_normalization
        metrics['el_distance']    = self.el_distance

        metrics['lifetime']    = self.lifetime

        # Mean responses:
        metrics['response_scale/pmt'] = tf.reduce_mean(self.pmt_response_scale)
        metrics['response_scale/pmt_var'] = tf.math.reduce_std(self.pmt_response_scale)

        metrics['response_scale/sipm'] = tf.reduce_mean(self.si_response_scale)
        metrics['response_scale/sipm_var'] = tf.math.reduce_std(self.si_response_scale)

        return metrics

    def regularization(self):

        # Hold the response scales close to 1, as they are a relative normalization:
        reg = tf.reduce_mean(tf.math.pow(self.si_response_scale - 1, 2))

        reg += tf.reduce_mean(tf.math.pow(self.pmt_response_scale - 1, 2))

        return reg

    # @tf.function
    def s2_call(self, inputs, diffusion_weight, training=True):

        responses = [ self.s2_subcall(i, d) for i, d in zip(inputs, diffusion_weight) ]

        pmt_responses, sipm_responses = zip(*responses)

        pmt_response  = tf.stack(pmt_responses)
        sipm_response = tf.stack(sipm_responses)

        # print("Final pmt_response.shape: ", pmt_response.shape)
        # print("Final sipm_response.shape: ", sipm_response.shape)

        return pmt_response, sipm_response


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

        exp_values = self.gaussian_norm * tf.exp( -(exp_input - z_ticks)**2 / (2*self.bin_sigma))

        #
        # Multiple by the weight of each electron (the lifetime weight):
        exp_values = tf.reshape(weight, weight.shape + (1,))*exp_values

        z_values_sparse = tf.sparse.from_dense(tf.transpose(exp_values))
        # print(z_values_sparse)

        # This runs the neural network to get the PMT and SiPM response:
        pmt_response  = self.s2pmt_call_network(xy_electrons)
        pmt_result  = tf.sparse.sparse_dense_matmul(z_values_sparse, pmt_response)
        pmt_result = tf.transpose(pmt_result)

        # For the sipm response, we are actually applying a
        # gaussian for the mean value of the response.
        


        # Here, we compute an gaussian point spread function
        # to represent the sipm response:

        xy_reshaped = tf.reshape(xy_electrons, (xy_electrons.shape[0], 1, 1, xy_electrons.shape[-1]))

        # For each electron, we subtract it's position from the sipm locations:
        r = tf.math.pow(xy_reshaped - self.sipm_locations, 2)

        total_distance = tf.pow(self.el_distance,2) + tf.reduce_sum(r, axis=-1)

        # We apply a factor of 1/r^2 to the light to each sipm.
        sipm_response = 1/(total_distance+1e-7) # Add a small threshold

        # gaussian_input = tf.reduce_sum(r, axis=-1)
        # gaussian_input = gaussian_input/ (2* tf.pow(self.sipm_sigma, 2))
        # sipm_response = tf.math.exp(-gaussian_input)

        # print(self.sipm_sigma)
        # sipm_norm = 1. / (tf.pow(self.sipm_sigma, 2) * 2 * 3.14159)

        # sipm_response *= sipm_norm
        # print(sipm_response.shape)
        # print("Mean response integral: ", tf.reduce_mean(tf.reduce_sum(sipm_response, axis=[1,2])))

        # The total sipm response, per electron, has to have an amplitude.
        # Regress this with a neural network
        sipm_amplitude  = self.sipm_response_normalization * self.s2si_call_network(xy_electrons)
        # print(sipm_amplitude.shape)
        # Reshape to make the broadcasting work:
        sipm_amplitude = tf.reshape(sipm_amplitude, (sipm_amplitude.shape[0], 1, 1))
        # print(sipm_response.shape)
        sipm_response *= sipm_amplitude


        # Here, we flatten the sipm_resonse matrix to enable the matmul:
        shape_cache = sipm_response.shape

        sipm_response = tf.reshape(sipm_response, (shape_cache[0], -1))

        # And, apply the individual sipm shape:
        sipm_response *= self.si_response_scale

        sipm_result = tf.sparse.sparse_dense_matmul(z_values_sparse, sipm_response)
        sipm_result = tf.transpose(sipm_result)
        sipm_result = tf.reshape(sipm_result, (shape_cache[1], shape_cache[2], self.n_ticks))



        return pmt_result, sipm_result

    # @profile
    @tf.function
    def s2pmt_call_network(self,xy_electrons):
        x = self.pmt_network(xy_electrons)

        return x
        # return x*self.pmt_response_scale


    @tf.function
    def s2si_call_network(self,xy_electrons):
        x = self.s2si_layer1(xy_electrons)
        x = self.s2si_layer2(x)
        
        return 1. + 0.5*x


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
            displacements[i] = self.diffusion_scale*displacements[i]*scale + mean
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

        probability = tf.math.exp( - z_position / self.lifetime)
        # probability = (1/self.lifetime) * tf.math.exp( - z_position / self.lifetime)

        return probability



    # @profile
    def apply_lifetime(self, diffused_electrons):

        # @profile


        selected_electrons = [ self.select_electrons(de) for de in diffused_electrons]

        return selected_electrons

    def call(self, energy_depositions, training=True):
        n_electrons, positions = self.generate_electrons(energy_depositions)
        diffused_electrons = self.diffuse_electrons(n_electrons, positions)
        diffusion_weight = self.apply_lifetime(diffused_electrons)
        s2pmt, s2si = self.s2_call(diffused_electrons, diffusion_weight)
        return s2pmt, s2si
