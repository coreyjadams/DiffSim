import jax.numpy as numpy

import flax.linen as nn

from dataclasses import dataclass
from . ElectronGenerator import ElectronGenerator, init_electron_generator
from . Diffusion import Diffusion
from . Lifetime import Lifetime
from . SensorResponse import SensorResponse, init_sensor_response

class NEW_Simulator(nn.Module):
# @dataclass
# class NEW_Simulator():

	eg: ElectronGenerator
	diff: Diffusion
	lifetime: Lifetime
	sr: SensorResponse

	@nn.compact
	def __call__(self, energies_and_positions):


		electrons, n_electrons = self.eg(energies_and_positions)

		diffused = self.diff(electrons)

		print(n_electrons)

		mask = self.lifetime(diffused, n_electrons)


		pmt_response = self.sr(diffused, mask)

		return pmt_response



def init_NEW_simulator(NEW_Physics):

	eg = init_electron_generator(NEW_Physics.electron_generator)

	sr = init_sensor_response(None)

	simulator = NEW_Simulator(
		eg   = eg,
		diff = Diffusion(),
		lifetime = Lifetime(),
		sr = sr,
	)

	return simulator