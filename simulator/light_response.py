import tensorflow as tf
import tensorflow_probability as tfp


class S1Pmt(tf.keras.models.Model):

    def __init__(self, n_sensors, waveform_length):
        tf.keras.models.Model.__init__(self)

        # Assume there is some latent space and map the input (x/y/z/E) to
        # the latent space:




        self.n_sensors = n_sensors
        self.latent_space = 12

        self.latent_map = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32), 
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(units=64), 
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(units=12*n_sensors),
            ]
        )

        self.upsample = tf.keras.Sequential([
            tf.keras.layers.UpSampling1D(size=2),
            tf.keras.layers.Conv1D(filters=self.n_sensors, kernel_size=10, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(filters=self.n_sensors, kernel_size=10, padding='same'),
            tf.keras.layers.LeakyReLU(),
            # tf.keras.layers.UpSampling1D(size=2),
            tf.keras.layers.Conv1D(filters=self.n_sensors, kernel_size=10, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(filters=self.n_sensors, kernel_size=10, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling1D(size=2),
            tf.keras.layers.Conv1D(filters=self.n_sensors, kernel_size=10, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(filters=self.n_sensors, kernel_size=10, padding='same'),
            tf.keras.layers.LeakyReLU(),
            ])

        # From here, we take the output feature vector and feed it
        # into unique sensor response simulators

        # Build it with 1D convolutions and upsampling


        self.final_layer = tf.keras.layers.Dense(
            units=n_sensors*waveform_length)

        self.s1_output_shape = (-1, n_sensors, waveform_length)

    def call(self, _input, training=True):

        # The _input should be Nx4 (x,y,z,energy)

        x = self.latent_map(_input)

        x = tf.reshape(x, (-1, self.latent_space, self.n_sensors))

        # Upsample
        x = self.upsample(x)


        # Here, the data is reshaped into 
        # [batch, 1, nfilters]

        
        x = tf.transpose(x, (0,2,1))

        return x


