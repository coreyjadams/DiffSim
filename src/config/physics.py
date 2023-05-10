from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import List


@dataclass
class ElectronGenerator:
    p1:  float = 22.4
    p2:  float = 0.15
    n_max: int = 2000   

@dataclass
class MLPConfig():
    layers:     List[int] = field(default_factory=lambda: [64, 128, 1])
    bias:            bool = True
    last_activation: bool = False


# @dataclass
# class DeepSetsCfg():
#     individual_cfg: MLPConfig = field(default_factory= lambda : MLPConfig(layers=[128,128]))
#     aggregate_cfg:  MLPConfig = field(default_factory= lambda : MLPConfig(layers=[128,1]))
#     active:              bool = True



@dataclass
class Simulator:
    detector: str = ""
    electron_generator: ElectronGenerator = ElectronGenerator() 

@dataclass
class NNSensorResponse:
    active:        bool = True
    mlp_cfg:  MLPConfig = field(default_factory= lambda : MLPConfig(layers =[64, 128]))
    waveform_ticks: int = 550
    bin_sigma:    float = 0.1
    n_sensors:      int = 12


@dataclass
class GSensorResponse:
    active:        bool = True
    mlp_cfg:  MLPConfig = field(default_factory= lambda : MLPConfig(layers =[64, 128, 1]))
    waveform_ticks: int = 550
    bin_sigma:    float = 0.1
    


@dataclass
class NEW_Simulator(Simulator):
    detector:            str = "NEW"
    pmt_s2: NNSensorResponse = field(default_factory = lambda : NNSensorResponse())
    sipm_s2: GSensorResponse = field(default_factory = lambda : GSensorResponse())

cs = ConfigStore.instance()
cs.store(group="physics", name="NEW_Simulator", node=NEW_Simulator)
