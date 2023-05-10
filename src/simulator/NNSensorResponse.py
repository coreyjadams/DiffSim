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
    sensor_simulator: MLP
    waveform_ticks:   int
    bin_sigma:        float

    # Functions to build waveforms based on weights and responses:
    @partial(vmap, in_axes=[None, 0,0,0])
    def build_waveforms(self, sensor_response, z_positions, weights):
        '''
        Compute the PMT response to electrons on the EL region
        '''

        n_electrons = z_positions.shape[0]
        print("n_electrons: ", n_electrons)
        # Build a range for the exponential input:
        starts = numpy.zeros(shape=(n_electrons)) # + 0.5
        stops  = numpy.ones(shape=(n_electrons)) * (self.waveform_ticks -1) # + 0.5

        # Reshape z positions for broadcasting:
        z_positions = z_positions.reshape((-1,1))

        exp_input = numpy.linspace(start=starts, stop=stops, num=self.waveform_ticks, axis=-1)
        print("exp_input.shape: ", exp_input.shape)

        exp_values = numpy.exp( - (exp_input - z_positions)**2.  / (2. * self.bin_sigma))
        print("exp_values.shape: ", exp_values.shape)


        # Scale by the weights:
        exp_values = exp_values * weights
        print("exp_values.shape: ", exp_values.shape)

        # Normalize the values:
        exp_values = exp_values.transpose() * (0.39894228040/numpy.sqrt(self.bin_sigma))
        print("weights.shape: ", weights.shape)

        print("exp_values.shape: ", exp_values.shape)

        # print("pmt exp_values.shape: ", exp_values.shape)
        # print("pmt sensor_response.shape: ", sensor_response.shape)
        waveforms = numpy.matmul(exp_values, sensor_response)
        print("waveforms.shape: ", waveforms.shape)
        return waveforms.transpose()

    @nn.compact
    def __call__(self, simulator_input, z_positions, mask):

        if self.active:
            print(self.sensor_simulator)
            response_of_sensors = self.sensor_simulator(simulator_input)


            waveforms = self.build_waveforms(response_of_sensors, z_positions, mask)

            return waveforms

def init_nnsensor_response(sensor_cfg):

    mlp_config = sensor_cfg.mlp_cfg
    mlp_config.layers.append(sensor_cfg.n_sensors)
    mlp, _ = init_mlp(mlp_config, nn.relu)

    # mlp = MLP(
    #     n_outputs  = [64, 12],
    #     bias       = True,
    #     activation = nn.relu,
    #     last_activation = True
    # )

    sr = NNSensorResponse(
        active           = sensor_cfg.active,
        sensor_simulator = mlp, 
        waveform_ticks   = sensor_cfg.waveform_ticks,
        bin_sigma        = sensor_cfg.bin_sigma
    )

    return sr, None