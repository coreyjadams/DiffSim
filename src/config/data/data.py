from enum import Enum
import pathlib

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class Data:
    name: str = MISSING

@dataclass
class Krypton(Data):
    name: str = "krypton"
    path: str = "/Users/corey.adams/data/NEXT/new_raw_data/"
    run: int  = 8678


cs = ConfigStore.instance()
cs.store(group="data", name="krypton", node=Krypton)