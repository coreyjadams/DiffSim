import jax.numpy as numpy
import jax.random as random
from jax import vmap

from dataclasses import dataclass
import flax.linen as nn

from functools import partial

from . MLP          import MLP, init_mlp
from . ConvLocalMLP import ConvLocalMLP, init_conv_local_mlp

def soft_exp(_x, a=8.):
    return numpy.exp(a * numpy.tanh(_x / a))


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
# 
    # Functions to build waveforms based on weights and responses:
    @partial(vmap, in_axes=[None, 0,0])
    def build_waveforms(self, sensor_response, z_positions):
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


        bin_sigma_v = self.variable(
                "nn_bin_sigma", "nn_bin_sigma",
                lambda s : 5*numpy.ones(s, dtype=z_positions.dtype),
                (1,), # shape is scalar
            )
        bin_sigma = bin_sigma_v.value



        exp_values = numpy.exp( - (exp_input - z_positions)**2.  / (2. * bin_sigma**2))


        # Normalize the values:
        # exp_values = exp_values.transpose()
        exp_values = exp_values * (0.39894228040/numpy.sqrt(bin_sigma**2))

        waveforms = numpy.matmul(sensor_response.T, exp_values)
        return waveforms

    @nn.compact
    def __call__(self, simulator_input, z_positions, mask):

        # input shape is (i_energy_dep, i_electron, -1)

        if self.active:

            raw_light_prob = self.el_light_prob(simulator_input)
            # The raw_light_prob should have the shape (N_energy_deps, N_electrons_max, n_sensors)


            # Put this through sigmoid to map from 0 to 1
            sensor_probs = nn.sigmoid(raw_light_prob)

            # Put this into exp to ensure >=0 and increase dynamic range.
            el_light_amp = self.el_light_amp(simulator_input) 
            
            # convert to a real amplitude, >= 0
            # The soft_exp function is like exp but prevents going arbitrarily high
            # el_light_amp   = soft_exp(el_light_amp)
            
            # The full response of the sensors is the product:
            # print("el_light_amp.shape: ", el_light_amp.shape)
            # print("sensor_probs.shape: ", sensor_probs.shape)
            response_of_sensors = el_light_amp * sensor_probs
            # print("response_of_sensors: ", response_of_sensors)
            # print("response_of_sensors.shape: ", response_of_sensors.shape)
            # print("simulator_input.shape: ", simulator_input.shape)
            # print("mask.shape: ", mask.shape)

            # Can probably apply the mask here instead of later:
            response_of_sensors = response_of_sensors * mask
            # print("z_positions.shape: ", z_positions.shape)
            # print("response_of_sensors.shape: ", response_of_sensors.shape)
            
            waveforms = self.build_waveforms(response_of_sensors, z_positions)
            
            # print(waveforms.shape)
            waveforms =  waveforms.sum(axis=0)
            # print(waveforms.shape)
  

            return waveforms

import copy

def init_nnsensor_response(sensor_cfg):

    mlp_config_sens = copy.copy(sensor_cfg.mlp_cfg)
    mlp_config_amp  = copy.copy(sensor_cfg.mlp_cfg)

    # mlp_config_sens.layers.append(sensor_cfg.n_sensors)
    
    # This MLP has N outputs (1 per sensor) and gets put into sigmoid
    # It represents the probability that light from this part of the EL
    # hits any particular sensor.
    # It's a conv_mlp meaning 

    mlp_sens, _ = init_conv_local_mlp(mlp_config_sens, sensor_cfg.n_sensors, nn.relu)

    # This MLP is the overall amount of light for this
    # region of the EL, and has only 1 output
    mlp_config_amp.layers.append(1)
    mlp_amp, _ = init_mlp(mlp_config_amp, nn.relu)


    sr = NNSensorResponse(
        active           = sensor_cfg.active,
        el_light_prob    = mlp_sens,
        el_light_amp     = mlp_amp,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma
    )

    return sr, None
