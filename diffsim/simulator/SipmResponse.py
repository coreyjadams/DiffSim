import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial, reduce

from .MLP import MLP, init_mlp

class SipmResponse(nn.Module):
    """
    Class to take in electrons at some locations and turn them into signals on sensors

    The MLP's final layer should have only 1 output (representing the amount of light produced)

    The output is the amount of light produced convolved with a gaussian into the sensor locations.

    So, the final axis will be the same shape as the sensor locations.

    """
    active:           bool
    psf_fn:           MLP
    waveform_ticks:   int
    bin_sigma:        float
    sensor_locations: numpy.ndarray


    # Functions to build waveforms based on weights and responses:
    # @partial(vmap, in_axes=[None, 0,0,0])
    def build_waveforms(self, emitted_photons, xy_positions, z_positions):
        '''
        Compute the sensor response to electrons on the EL region, with a guassian spread
        '''

        print("emitted_photons.shape: ", emitted_photons.shape)
        print("xy_positions.shape: ", xy_positions.shape)
        print("z_positions.shape: ", z_positions.shape)

        sensor_shape = self.sensor_locations.shape[0:2]
        n_sensors    = reduce(lambda x, y : x*y, sensor_shape, 1)

        # Reshape z positions for broadcasting:
        z_positions = z_positions.reshape((-1,1))
        _xy_reshaped = xy_positions.reshape((xy_positions.shape[0], 1,1,xy_positions.shape[-1]))

        subtracted_values = _xy_reshaped - self.sensor_locations
        r_squared = (subtracted_values**2).sum(-1)
        r = numpy.sqrt(r_squared)

        # psf_input = numpy.stack([r_squared, r,  1./r, 1./r_squared ], axis=-1)
        # # print("psf_input.shape: ", psf_input.shape)
        # From r and r**2, compute the response of all sipms to each emitted photon
        # Units of pe / photon
        baseline = numpy.exp( - 0.01*(r**2))

        r = r / 500.
        psf_fn_output = self.psf_fn(r)

        # Normalizing the point spread function with some physics-based priors:


        amplitude_v = self.variable(
            "params", "amplitude",
            lambda s : -1e1*numpy.ones(s, dtype=r.dtype),
            (1,), # shape is scalar
        )
        amplitude = numpy.exp(amplitude_v.value)

        psf_output = amplitude * baseline * (1 + psf_fn_output)
        # psf_output = nn.sigmoid(psf_fn_output - 0.1*r_squared.reshape(psf_fn_output.shape))
        # print("psf_output.shape: ", psf_output.shape)
        # print("emitted_photons.shape: ", emitted_photons.shape)

        sensor_response = emitted_photons * psf_output.reshape((-1, n_sensors))

        # Build a range for the exponential input:
        n_electrons = z_positions.shape[0]
        starts = numpy.zeros(shape=(n_electrons)) # + 0.5
        stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5

        exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)

        # bin_sigma_v = self.variable(
        #         "nn_bin_sigma", "nn_bin_sigma",
        #         lambda s : 0.1*numpy.ones(s, dtype=z_positions.dtype),
        #         (1,), # shape is scalar
        #     )
        # bin_sigma = bin_sigma_v.value

        # Force this value to be between 0 and 1!  (With a floor at 0.05)
        # bin_sigma = 0.05 + nn.sigmoid(self.bin_sigma)

        exp_values = numpy.exp( - (exp_input - z_positions )**2.  / (2. * self.bin_sigma))

        # Normalize the values:
        exp_values = exp_values.transpose() * (0.39894228040/numpy.sqrt(self.bin_sigma))

        waveforms = numpy.matmul(exp_values, sensor_response)
        # And, unflatten:
        waveforms = waveforms.reshape((-1, *sensor_shape))

        return waveforms.transpose((1,2,0))
    

    @nn.compact
    def __call__(self, el_photons, xy_positions, z_positions):

        if self.active:


            emitted_photons = el_photons

            # Turn the photons into waveforms:
            waveforms = self.build_waveforms(
                emitted_photons, xy_positions, z_positions)


            waveforms = waveforms.sum(axis=0)

            # print(waveforms.shape)
            shape = waveforms.shape
            sensor_shape = self.sensor_locations.shape[0:2]

            waveforms = waveforms.reshape(sensor_shape + (shape[-1],))

            # sensor_scale = self.variable(
            #     "params", "sensor_scale",
            #     lambda s : 1e-1*numpy.ones(s, dtype=r.dtype),
            #     self.sensor_locations.shape, # shape is scalar
            # )
            # print(waveforms.shape)
            # exit()

            # # The waveforms are scaled overall by a parameter _per sensor_:
            # sensor_shape = self.sensor_locations.shape[0:2]
            # waveform_scale_v = self.variable(
            #     "waveform_scale", "waveform_scale",
            #     lambda s : 1.0*numpy.ones(s, dtype=waveforms.dtype),
            #     sensor_shape
            # )
            # waveform_scale = waveform_scale_v.value
            # waveforms = waveforms * (1. + waveform_scale.reshape(sensor_shape + (-1,)))

            return waveforms
        else:
            return None

def init_sipm_response(sensor_cfg):


    # The sipm locations:
    sipms_1D = numpy.arange(-240, 240, 10.) + 5
    n_sipms = sipms_1D.shape[0]
    sipm_locations_x = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms))
    sipm_locations_y = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms)).transpose()

    sipm_locations = numpy.stack([sipm_locations_y, sipm_locations_x], -1)


    mlp_config = sensor_cfg.mlp_cfg
    
    # n_sipms = 48*48; mlp_config.layers.append(n_sipms)
    mlp, _ = init_mlp(mlp_config, nn.sigmoid)


    sr = SipmResponse(
        active           = sensor_cfg.active,
        psf_fn           = mlp,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma,
        sensor_locations = sipm_locations,
    )

    # Return None for the rng parameters
    return sr, None
