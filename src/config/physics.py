from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ElectronGenerator:
    p1:  float = 22.4
    p2:  float = 0.15
    n_out: int = 2000   


@dataclass
class Physics:
    detector: str = ""
    electron_generator: ElectronGenerator = ElectronGenerator() 


@dataclass
class NEW_Physics(Physics):
    detector: str = "NEW"


cs = ConfigStore.instance()
cs.store(group="physics", name="NEW_Physics", node=NEW_Physics)
