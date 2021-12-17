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
        self.diffusion_scale = tf.Variable([0.5, 0.5, 0.1])
        
        self.electron_normal_distribution = tfp.distributions.Normal(
            loc = 0.0, 
            scale = 1.0
        )
        
        self.s2pmt_layer1 = tf.keras.layers.Dense(units=28, activation="sigmoid")
        self.s2pmt_layer2 = tf.keras.layers.Dense(units=12, activation="sigmoid")
        
        # PMT Response scale:
        self.pmt_response_scale = tf.Variable(tf.ones(shape=[12]))
        
        # Lifetime variables:
        # self.lifetime = tf.Variable(25.)
        # self.uniform_sampler = tfp.distributions.Uniform()
        
        
    def s2pmt_call(self, inputs, training=True):

        response = tf.stack([ self.s2pmt_subcall(d) for d in inputs ] )
        return response
    
    def s2pmt_subcall(self, electrons):
        # Pull out z_ticks:
        z_ticks = electrons[:,2]
        # Get the x/y locations
        xy_electrons = electrons[:,0:2]
        
        # Reshape the ticks:
        z_ticks = tf.reshape(z_ticks, z_ticks.shape + (1,))
        # Stack the tick ranges, one per tick:
        exp_input = tf.stack( [tf.range(self.n_ticks,dtype=tf.dtypes.float32) for _z in z_ticks ])
        # Apply the exponential, transpose, and make sparse:
        z_values_sparse = tf.sparse.from_dense(tf.transpose(tf.exp( -(exp_input - z_ticks)**2 / 0.1)))
        # print(z_values_sparse)
        
        # This runs the neural network to get the PMT response:
        pmt_response = self.s2pmt_call_network(xy_electrons)
        result = tf.sparse.sparse_dense_matmul(z_values_sparse, pmt_response)
        result = tf.transpose(result)
        


        return result
        
    def s2pmt_call_network(self,xy_electrons):
        x = self.s2pmt_layer1(xy_electrons)
        x = self.s2pmt_layer2(x)
        return x*tf.math.pow(self.pmt_response_scale, 2)
    
    
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
    
    def apply_lifetime(self, electrons):

        print(electrons.shape)
        z_position = electrons[:,2]
        # This function probabilistically removes electrons based upon the lifetime.
        probability = 1 - tf.math.exp( - electrons[:,2] / self.lifetime)
        accepted = probability > self.uniform_sampler.sample(len(electrons))
        selected_electrons = tf.boolean_mask(electrons, accepted)
        return selected_electrons
    
    def call(self, energy_depositions):
        n_electrons, positions = self.generate_electrons(energy_depositions)    
        diffused_electrons = self.diffuse_electrons(n_electrons, positions)
        s2pmt = self.s2pmt_call(diffused_electrons)
        return s2pmt