import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial, reduce

from .MLP import MLP, init_mlp

class SipmSensorResponse(nn.Module):
    """
    Class to take in electrons at some locations and turn them into signals on sensors

    The MLP's final layer should have only 1 output (representing the amount of light produced)

    The output is the amount of light produced convolved with a gaussian into the sensor locations.

    So, the final axis will be the same shape as the sensor locations.

    """
    active:           bool
    sensor_simulator: MLP
    waveform_ticks:   int
    bin_sigma:        float
    sensor_locations: numpy.ndarray


    # Functions to build waveforms based on weights and responses:
    @partial(vmap, in_axes=[None, 0,0,0])
    def build_waveforms(self, sensor_response, z_positions, weights):
        '''
        Compute the PMT response to electrons on the EL region
        '''

        n_electrons = z_positions.shape[0]
        # Build a range for the exponential input:
        starts = numpy.zeros(shape=(n_electrons)) # + 0.5
        stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5

        # Reshape z positions for broadcasting:
        z_positions = z_positions.reshape((-1,1))

        exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)

        exp_values = numpy.exp( - (exp_input - z_positions)**2.  / (2. * self.bin_sigma))

        # Scale by the weights:
        exp_values = exp_values * weights

        # Normalize the values:
        exp_values = exp_values.transpose() * (0.39894228040/numpy.sqrt(self.bin_sigma))

        waveforms = numpy.matmul(exp_values, sensor_response)

        return waveforms.transpose()


    @nn.compact
    def __call__(self, simulator_input, z_positions, mask):

        if self.active:

            # The sensor simulator represents the total amount of light emitted
            # at this particular point on the EL region.
            response_of_sensors = self.sensor_simulator(simulator_input)
            # The exp forces it to be positive and gives a broad dynamic range:
            response_of_sensors = numpy.exp(response_of_sensors)

            waveforms = self.build_waveforms(
                response_of_sensors, z_positions, mask)


            waveforms = waveforms.sum(axis=0)

            # print(waveforms.shape)
            shape = waveforms.shape
            waveforms = waveforms.reshape((47,47) + (shape[-1],))

            print(numpy.max(waveforms))

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

def init_sipm_sensor_response(sensor_cfg):


    # The sipm locations:
    sipms_1D = numpy.arange(-235, 235, 10.) + 5
    n_sipms = sipms_1D.shape[0]
    sipm_locations_x = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms))
    sipm_locations_y = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms)).transpose()

    sipm_locations = numpy.stack([sipm_locations_y, sipm_locations_x], -1)

    n_sipms = 47*47

    mlp_config = sensor_cfg.mlp_cfg
    mlp_config.layers.append(n_sipms)
    print(mlp_config)
    mlp, _ = init_mlp(mlp_config, nn.sigmoid)


    sr = SipmSensorResponse(
        active           = sensor_cfg.active,
        sensor_simulator = mlp,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma,
        sensor_locations = sipm_locations,
    )

    # Return None for the rng parameters
    return sr, None
