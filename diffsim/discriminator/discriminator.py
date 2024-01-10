import jax.numpy as numpy

import flax.linen as nn

from dataclasses import dataclass

from typing import Tuple

class Block(nn.Module):
    residual: bool
    features: int
    kernel:   Tuple[int]

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(self.features, self.kernel, use_bias=False)(inputs)

        x = nn.LayerNorm()(x)

        if self.residual:
            x = nn.activation.celu(x)

            x = nn.Conv(self.features, self.kernel, use_bias=False)(x)

            x = nn.LayerNorm()(x)

        x = nn.activation.celu(x)

        return x



class DiscriminatorSipm(nn.Module):

    layers: int

    @nn.compact
    def __call__(self, inputs):

        s = inputs.shape
        inputs = inputs.reshape(s + (1,))

        n_filters = 8
        x = nn.Conv(n_filters, [6,6,25], strides=[6,6,25], padding='VALID')(inputs)


        for i_layer in range(self.layers):
            x = Block(False, n_filters, [3,3,6])(x)
            x = nn.max_pool(x, (2,2,4), (2,2,4))
            # n_filters += 8

        return nn.avg_pool(x, x.shape[0:-1]).reshape((-1,))
        return x




class DiscriminatorPMT(nn.Module):

    layers: int

    @nn.compact
    def __call__(self, inputs):
        s = inputs.shape
        inputs = inputs.reshape(s + (1,))

        n_filters = 8
        x = nn.Conv(n_filters, [1,15,], use_bias=False)(inputs)


        for i_layer in range(self.layers):
            x = Block(False, n_filters, [1,5,],)(x)
            x = nn.max_pool(x, (1,4,), (1,4,))
            # n_filters += 8

        return nn.max_pool(x, x.shape[0:-1]).reshape((-1,))
    
class Discriminator(nn.Module):

    sipm: DiscriminatorSipm
    pmt: DiscriminatorPMT

    @nn.compact
    def __call__(self, inputs):

        pmt_sig  = self.pmt( inputs["S2Pmt"])
        # sipm_sig = self.sipm(inputs["S2Si"])


        # merged = numpy.concatenate([pmt_sig, sipm_sig])
        # final_prediction = nn.Dense(1)(merged)
        
        final_prediction = nn.Dense(1, use_bias=False)(pmt_sig)

        return final_prediction.reshape()


def init_discriminator_model():


    disc = Discriminator(
        DiscriminatorSipm(2),
        DiscriminatorPMT(3),
    )

    return disc
