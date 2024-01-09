import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial, reduce

from .MLP import MLP, init_mlp

class ConvSensorRespons(nn.Module):
    """
    Class to take in electrons at some locations and turn them into signals on a sensor grid

    The network goes in one pass with a convNN to turn an x/y position into the probabilty
    of photons hitting a particular sipm.

    An MLP predicts the amplitude of light (Actually the log(amp)) from that position

    So, the final axis will be the same shape as the sensor locations.

    """
    active:           bool
    el_simulator:     MLP
    sipm_simulator:   ConvNN
    waveform_ticks:   int
    bin_sigma:        float
    sensor_locations: numpy.ndarray


    # Functions to build waveforms based on weights and responses:
    @partial(vmap, in_axes=[None, 0,0,0,0])
    def build_waveforms(self, sensor_response, xy_positions, z_positions, weights):
        '''
        Compute the sensor response to electrons on the EL region, with a guassian spread
        '''

        sensor_shape = self.sensor_locations.shape[0:2]
        n_sensors    = reduce(lambda x, y : x*y, sensor_shape, 1)

        # Reshape z positions for broadcasting:
        z_positions = z_positions.reshape((-1,1))
        _xy_reshaped = xy_positions.reshape((xy_positions.shape[0], 1,1,xy_positions.shape[-1]))

        subtracted_values = _xy_reshaped - self.sensor_locations

        r_squared = (subtracted_values**2).sum(-1)

        el_spread_v = self.variable(
                "el_spread", "el_spread",
                lambda s : 0.1*numpy.ones(s, dtype=sensor_response.dtype),
                (1,),
            )
        # This actually fetches the value:
        el_spread = el_spread_v.value

        # Run the subtracted values through a gaussian response:
        sipm_spread_response = numpy.exp(-0.5*(r_squared/(el_spread)**2) ) / (el_spread * 2.5066)


        sensor_response = sipm_spread_response * sensor_response.reshape((-1,1,1))


        # Multiple the total light (sensor response) by the 1/r^2 scaling:

        # We put the subtracted differences through a 1/r^2 response

        # Build a range for the exponential input:
        n_electrons = z_positions.shape[0]
        starts = numpy.zeros(shape=(n_electrons)) # + 0.5
        stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5

        exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)
        exp_values = numpy.exp( - (exp_input - z_positions)**2.  / (2. * self.bin_sigma))

        # Normalize the values:
        exp_values = exp_values.transpose() * (0.39894228040/numpy.sqrt(self.bin_sigma))
        # Scale by the weights:
        exp_values = exp_values * weights.T

        # To do the matmul, we have to flatten the sensor_response briefly
        _sensor_response_flat = sensor_response.reshape((-1, n_sensors))
        waveforms = numpy.matmul(exp_values, _sensor_response_flat)

        # And, unflatten:
        waveforms = waveforms.reshape((-1, *sensor_shape))
        return waveforms.transpose((1,2,0))




    @nn.compact
    def __call__(self, simulator_input, z_positions, mask):

        if self.active:
            # The sensor simulator represents the total amount of light emitted
            # at this particular point on the EL region.
            response_of_sensors = self.sensor_simulator(simulator_input)
            # The exp forces it to be positive and gives a broad dynamic range:
            response_of_sensors = numpy.exp(response_of_sensors)

            waveforms = self.build_waveforms(
                response_of_sensors, simulator_input, z_positions, mask)

            waveforms = waveforms.sum(axis=0)

            # The waveforms are scaled overall by a parameter _per sensor_:
            sensor_shape = self.sensor_locations.shape[0:2]
            waveform_scale_v = self.variable(
                "waveform_scale", "waveform_scale",
                lambda s : 1.0*numpy.ones(s, dtype=waveforms.dtype),
                sensor_shape
            )
            waveform_scale = waveform_scale_v.value
            waveforms = waveforms * (1. + waveform_scale.reshape(sensor_shape + (-1,)))

            return waveforms
        else:
            return None

def init_gsensor_response(sensor_cfg):

    mlp_config = sensor_cfg.mlp_cfg
    mlp, _ = init_mlp(mlp_config, nn.sigmoid)

    # The sipm locations:
    sipms_1D = numpy.arange(-235, 235, 10.) + 5
    n_sipms = sipms_1D.shape[0]
    sipm_locations_x = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms))
    sipm_locations_y = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms)).transpose()

    sipm_locations = numpy.stack([sipm_locations_y, sipm_locations_x], -1)

    sr = ConvSensorRespons(
        active           = sensor_cfg.active,
        sensor_simulator = mlp,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma,
        sensor_locations = sipm_locations,
    )

    # Return None for the rng parameters
    return sr, None
