import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial

from .MLP import MLP, init_mlp

class NNSensorResponse(nn.Module):
    """
    Class to take in electrons at some locations and turn them into signals on sensors

    The MLP's final layer should be the number of sensors desired.

    """
    active:           bool
    el_light_prob:    MLP # NN that simulates the probability of light from EL position hitting sensor
    el_light_amp:     MLP # NN that simulates the amount of light from EL position overall
    waveform_ticks:   int
    bin_sigma:        float

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
            # Put this through sigmoid to map from 0 to 1:
            sensor_probs = nn.sigmoid(self.el_light_prob(simulator_input))
            # Put this into exp to ensure >=0 and increase dynamic range.

            # We compute the log of the light response amplitude from the NN
            # Further, the assumption is that the amount of light hitting a sensor
            # Is approximately constant.  So we predict the constant + a position-
            # dependant correction

            sensor_amp   = numpy.exp(self.el_light_amp(simulator_input) )

            response_of_sensors = sensor_amp * sensor_probs

            waveforms = self.build_waveforms(response_of_sensors, z_positions, mask)

            waveforms =  waveforms.sum(axis=0)

            # # The waveforms are scaled overall by a parameter:
            # waveform_scale_v = self.variable(
            #     "waveform_scale", "waveform_scale",
            #     lambda s : 100.*numpy.ones(s, dtype=waveforms.dtype),
            #     (1,), # shape is scalar
            # )
            # waveform_scale = waveform_scale_v.value
            # waveforms = waveforms * waveform_scale

            return waveforms

import copy

def init_nnsensor_response(sensor_cfg):

    mlp_config_sens = copy.copy(sensor_cfg.mlp_cfg)
    mlp_config_amp  = copy.copy(sensor_cfg.mlp_cfg)

    mlp_config_sens.layers.append(sensor_cfg.n_sensors)
    # This MLP has 12 outputs and gets put into sigmoid
    # It represents the probability that light from this part of the EL
    # Hits any particular sensor.
    mlp_sens, _ = init_mlp(mlp_config_sens, nn.relu)

    # This MLP is the overall amount of light for this
    # region of the EL, and has only 1 output
    mlp_config_amp.layers.append(1)
    mlp_amp, _ = init_mlp(mlp_config_amp, nn.relu)


    sr = NNSensorResponse(
        active           = sensor_cfg.active,
        el_light_prob    = mlp_amp,
        el_light_amp     = mlp_sens ,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma
    )

    return sr, None
