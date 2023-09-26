import jax
from jax import numpy

import flax.linen as nn

from typing import Tuple


# This is a module in flax that we'll use to build up the bigger modules:
class ConvLocalMLP(nn.Module):
    n_outputs:       Tuple[int, ...]
    bias:            bool
    activation:      callable
    last_activation: bool
    n_sensors:       int

    # def setup(self):

    #     self.layers = [
    #         nn.ConvLocal(n_out,
    #             use_bias = self.bias,
    #             # kernel_init = nn.initializers.xavier_uniform(),
    #             # bias_init   = nn.initializers.constant(10.)
    #             )
    #         for n_out in self.n_outputs
    #     ]

    @nn.compact
    def __call__(self, x):


        s = x.shape
        # Add an extra dimension before the last dimension:
        new_shape = s[:-1] + (1, s[-1])
        # print(x.shape)

        # Put the sensors in the second to last dim:
        prob_sensor_input = numpy.repeat(x.reshape(new_shape), self.n_sensors, axis=-2)
        # print("Repeated sensor input: ", prob_sensor_input.shape  )

        layer_input =prob_sensor_input

        # We need to reshape the input, x, into a suitable image shape
        # print("layer_input.shape: ", layer_input.shape, flush=True)


        # Loop over the layers
        for i, output_size in enumerate(self.n_outputs):
            # compute the application of the layer:
            layer_output = nn.ConvLocal(output_size, kernel_size=[1,], use_bias=True)(layer_input)
            # print(i, "Output shape: ", layer_output.shape, flush=True)
            # print(numpy.mean(layer_output[0,:,0,:],))
            # print(numpy.mean(layer_output[0,:,1,:],))
            # If it's the last layer, don't apply activation if not specified:

            if i != len(self.n_outputs) - 1 or self.last_activation:
                layer_output = self.activation(layer_output)

            # Prepare for the next layer:
            layer_input = layer_output

        # print("Output shape: ", layer_output.shape, flush=True)
        # Before returning, we have to remove the extra dimension:
        # print(layer_output[0,0:10,0,:],)
        # print(layer_output[0,0:10,1,:],)
        # exit()
        layer_output = layer_output.reshape(layer_output.shape[:-1])
        return layer_output
    


def init_conv_local_mlp(mlp_cfg, n_sensors, activation):

    mlp = ConvLocalMLP(
        n_outputs       = mlp_cfg.layers,
        bias            = mlp_cfg.bias,
        activation      = activation,
        last_activation = mlp_cfg.last_activation,
        n_sensors       = n_sensors,
    )

    return mlp, None
