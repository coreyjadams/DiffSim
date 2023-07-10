import jax.numpy as numpy

import flax.linen as nn

from dataclasses import dataclass
from . ElectronGenerator import ElectronGenerator, init_electron_generator
from . Diffusion         import Diffusion,         init_diffusion
from . Lifetime          import Lifetime,          init_lifetime
from . NNSensorResponse  import NNSensorResponse,  init_nnsensor_response
# from . SipmResponse   import SipmSensorResponse,   init_sipm_sensor_response
from . GSensorResponse   import GSensorResponse,   init_gsensor_response

class NEW_Simulator(nn.Module):

    eg:       ElectronGenerator
    diff:     Diffusion
    lifetime: Lifetime
    pmt_s2:   NNSensorResponse
    sipm_s2:  GSensorResponse

    @nn.compact
    def __call__(self, energies_and_positions):

        electrons, n_electrons = self.eg(energies_and_positions)


        diffused = self.diff(electrons)

        mask = self.lifetime(diffused, n_electrons)


        # pmts only depend on xy:
        diffused_xy = diffused[:,:,0:2]
        diffused_z  = diffused[:,:,2]

        pmt_response = self.pmt_s2(diffused_xy, diffused_z, mask)

        sipm_response = self.sipm_s2(diffused_xy, diffused_z, mask)


        return {
            # "N_e"   : n_electrons,
        	"S2Pmt" : pmt_response,
        	"S2Si"  : sipm_response
    	}



def init_NEW_simulator(NEW_Physics):

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

    simulator = NEW_Simulator(
        eg       = eg,
        diff     = diff,
        lifetime = Lifetime(),
        pmt_s2   = pmt_s2,
        sipm_s2  = sipm_s2,
    )


    return simulator, all_rng_keys
