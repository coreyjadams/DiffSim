import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial, reduce

from .MLP import MLP, init_mlp

# @jit
def soft_exp(_x, a=8.):
    return numpy.exp(a * numpy.tanh(_x / a))



class GSensorResponse(nn.Module):
    """
    Class to take in electrons at some locations and turn them into signals on sensors

    The MLP's final layer should have only 1 output (representing the amount of light produced)

    The output is the amount of light produced convolved with a gaussian into the sensor locations.

    So, the final axis will be the same shape as the sensor locations.

    """
    active:           bool
    EL_simulator: MLP
    waveform_ticks:   int
    bin_sigma:        float
    sensor_locations: numpy.ndarray


    # Functions to build waveforms based on weights and responses:
    @partial(vmap, in_axes=[None, 0,0,0])
    def build_waveforms(self, sensor_response, xy_positions, z_positions):
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
                lambda s : numpy.ones(s, dtype=sensor_response.dtype),
                (1,),
            )

        # This actually fetches the value:
        el_spread = el_spread_v.value
        # Square the variance to ensure its positive:
        variance = el_spread**2
        # Run the subtracted values through a gaussian response:
        sipm_spread_response = numpy.exp(-0.5*(r_squared/variance) )

        # Normalize the response:
        sipm_spread_response = sipm_spread_response * (1./(2*3.14159*variance))



        sensor_response = sipm_spread_response * sensor_response.reshape((-1,1,1))

        # Build a range for the exponential input:
        n_electrons = z_positions.shape[0]
        starts = numpy.zeros(shape=(n_electrons)) # + 0.5
        stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5


        bin_sigma_v = self.variable(
                "nn_bin_sigma", "nn_bin_sigma",
                lambda s : numpy.ones(s, dtype=z_positions.dtype),
                (1,), # shape is scalar
            )
        bin_sigma = bin_sigma_v.value
        # Square to ensure > 0:
        bin_sigma = bin_sigma**2

        exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)
        # I don't know why - the sipm data is shifted in Z compared to pmt
        exp_values = numpy.exp( - (exp_input - z_positions )**2.  / (2. * bin_sigma))

        # Normalize the values:
        exp_values = exp_values * (0.39894228040/numpy.sqrt(bin_sigma))
        # exp_values = exp_values.transpose() * (0.39894228040/numpy.sqrt(bin_sigma))
        # Scale by the weights:


        # To do the matmul, we have to flatten the sensor_response briefly
        _sensor_response_flat = sensor_response.reshape((-1, n_sensors))


        waveforms = numpy.matmul(_sensor_response_flat.T, exp_values)

        # And, unflatten:
        waveforms = waveforms.reshape((*sensor_shape, -1))

        return waveforms




    @nn.compact
    def __call__(self, simulator_input, z_positions, mask):

        if self.active:

            # The EL simulator represents the total amount of light emitted
            # at this particular point on the EL region.
            response_of_sensors = self.EL_simulator(simulator_input)
            # The exp forces it to be positive and gives a broad dynamic range:
            response_of_sensors = numpy.exp(response_of_sensors)

            response_of_sensors = response_of_sensors * mask

            waveforms = self.build_waveforms(
                response_of_sensors, simulator_input, z_positions)

            waveforms = waveforms.sum(axis=0)


            # # The waveforms are scaled overall by a parameter _per sensor_:
            # sensor_shape = self.sensor_locations.shape[0:2]
            # waveform_scale_v = self.variable(
            #     "waveform_scale", "waveform_scale",
            #     lambda s : 0.1*numpy.ones(s, dtype=waveforms.dtype),
            #     sensor_shape
            # )
            # waveform_scale = waveform_scale_v.value
            # waveforms = waveforms * (1. + waveform_scale.reshape(sensor_shape + (-1,)))

            return waveforms
        else:
            return None

def init_gsensor_response(sensor_cfg):

    mlp_config = sensor_cfg.mlp_cfg
    mlp_config.last_activation=True
    mlp, _ = init_mlp(mlp_config, nn.relu)

    # The sipm locations:
    # sipms_1D = numpy.arange(-235, 235, 10.) + 5
    sipms_1D = numpy.arange(-240, 240, 10.) + 5
    # print(sipms_1D)
    n_sipms = sipms_1D.shape[0]
    # print(n_sipms)

    # sipms_1D = numpy.arange(-235, 235, 10.) + 5
    # print(sipms_1D)
    # n_sipms = sipms_1D.shape[0]
    # print(n_sipms)
    sipm_locations_x = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms))
    sipm_locations_y = numpy.tile(sipms_1D, (n_sipms,)).reshape((n_sipms, n_sipms)).transpose()

    sipm_locations = numpy.stack([sipm_locations_y, sipm_locations_x], -1)

    sr = GSensorResponse(
        active           = sensor_cfg.active,
        EL_simulator     = mlp,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma,
        sensor_locations = sipm_locations,
    )

    # Return None for the rng parameters
    return sr, None
