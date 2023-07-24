import jax.numpy as numpy

import flax.linen as nn

from dataclasses import dataclass

class Discriminator(nn.Module):

    @nn.compact
    def __call__(self, waveforms):
        
        


def init_discriminator(NEW_Physics):

    all_rng_keys = []

    eg, eg_rng_keys = init_electron_generator(NEW_Physics.electron_generator)

    if eg_rng_keys is not None:
        all_rng_keys += eg_rng_keys

    diff, diff_rng_keys = init_diffusion(NEW_Physics.diffusion)
    if diff_rng_keys is not None:
        all_rng_keys += diff_rng_keys


    lifetime, lifetime_rng_keys = init_lifetime()
    if lifetime_rng_keys is not None:
        all_rng_keys += lifetime_rng_keys


    pmt_s2, _ = init_nnsensor_response(NEW_Physics.pmt_s2)
    # sipm_s2, _ = init_sipm_sensor_response(NEW_Physics.sipm_s2)
    sipm_s2, _ = init_gsensor_response(NEW_Physics.sipm_s2)

    simulator = Discriminator(
        eg       = eg,
        diff     = diff,
        lifetime = Lifetime(),
        pmt_s2   = pmt_s2,
        sipm_s2  = sipm_s2,
    )


    return simulator, all_rng_keys
