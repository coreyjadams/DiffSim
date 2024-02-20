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
    sens_response:    MLP # NN that simulates the probability of light from EL position hitting sensor
    waveform_ticks:   int
    bin_sigma:        float
# 
    # Functions to build waveforms based on weights and responses:
    # @partial(vmap, in_axes=[None, 0,0])
    def build_waveforms(self, sensor_response, z_positions):
        '''
        Compute the PMT response to electrons on the EL region

        This function operates on a single electron!

        '''
        # print("sensor_response.shape: ", sensor_response.shape)
        # print("z_positions.shape: ", z_positions.shape)
        n_electrons = z_positions.shape[0]
        # Build a range for the exponential input:
        starts = numpy.zeros(shape=(n_electrons)) # + 0.5
        stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5

        # Reshape z positions for broadcasting:
        z_positions = z_positions.reshape((-1,1))

        exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)
        # print("exp_input.shape: ", exp_input.shape)

        # bin_sigma_v = self.variable(
        #         "nn_bin_sigma", "nn_bin_sigma",
        #         lambda s : 0.1*numpy.ones(s, dtype=z_positions.dtype),
        #         (1,), # shape is scalar
        #     )
        # bin_sigma = bin_sigma_v.value

        # # Force this value to be between 0 and 1!  (With a floor at 0.05)
        # bin_sigma = 0.05 + nn.sigmoid(bin_sigma)

        bin_sigma = self.bin_sigma

        exp_values = numpy.exp( - (exp_input - z_positions)**2.  / (2. * bin_sigma**2))
        # print("exp_values.shape: ", exp_values.shape)

        # Normalize the values:
        # exp_values = exp_values.transpose()
        exp_values = exp_values * (0.39894228040/numpy.sqrt(bin_sigma**2))

        waveforms = numpy.matmul(sensor_response.T, exp_values)
        return waveforms
    

    # # Functions to build waveforms based on weights and responses:
    # # @partial(vmap, in_axes=[None, 0,0])
    # def build_waveforms(self, sensor_responses, z_position):
    #     '''
    #     Compute the PMT response to electrons on the EL region

    #     This function operates on a single electron and all sensors

    #     '''
    #     print("sensor_response.shape: ", sensor_response.shape)
    #     print("z_positions.shape: ", z_positions.shape)

    #     n_sensors = 
    #     # # Build a range for the exponential input:
    #     starts = numpy.zeros(shape=(n_electrons)) # + 0.5
    #     # stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5

    #     # # Reshape z positions for broadcasting:
    #     # z_positions = z_positions.reshape((-1,1))

    #     # exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)

    #     exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)

    #     # bin_sigma_v = self.variable(
    #     #         "nn_bin_sigma", "nn_bin_sigma",
    #     #         lambda s : 0.1*numpy.ones(s, dtype=z_positions.dtype),
    #     #         (1,), # shape is scalar
    #     #     )
    #     # bin_sigma = bin_sigma_v.value

    #     # # Force this value to be between 0 and 1!  (With a floor at 0.05)
    #     # bin_sigma = 0.05 + nn.sigmoid(bin_sigma)

    #     bin_sigma = self.bin_sigma

    #     exp_values = numpy.exp( - (exp_input - z_positions)**2.  / (2. * bin_sigma**2))


    #     # Normalize the values:
    #     # exp_values = exp_values.transpose()
    #     exp_values = exp_values * (0.39894228040/numpy.sqrt(bin_sigma**2))

    #     waveforms = numpy.matmul(sensor_response.T, exp_values)
    #     return waveforms

    @nn.compact
    def __call__(self, el_photons, xy_positions, z_positions):

        # input shape is (i_energy_dep, i_electron, -1)

        if self.active:

            response = self.sens_response(xy_positions)
            print("el_photons.shape: ", el_photons.shape)
            print("response.shape: ", response.shape)
            # The pmt_response should have the shape (N_energy_deps, N_electrons_max, n_sensors)


            # Put this through exp to map from 0 to 1
            sensor_probs = nn.sigmoid(response)

            
            # The full response of the sensors is the product:
            response_of_sensors = el_photons * sensor_probs

            
            waveforms = self.build_waveforms(response_of_sensors, z_positions)
            
            print("waveforms.shape: ", waveforms.shape)

            # waveforms =  waveforms.sum(axis=0)
            
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

    # mlp_config_sens.layers.append(sensor_cfg.n_sensors)
    
    # This MLP has N outputs (1 per sensor) and gets put into sigmoid
    # It represents the probability that light from this part of the EL
    # hits any particular sensor.
    # It's a conv_mlp meaning 

    mlp_sens, _ = init_conv_local_mlp(mlp_config_sens, sensor_cfg.n_sensors, nn.relu)



    sr = NNSensorResponse(
        active           = sensor_cfg.active,
        sens_response    = mlp_sens,
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma
    )

    return sr, None
