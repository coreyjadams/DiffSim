from enum import Enum
import pathlib

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING


class RandomMode(Enum):
    random_blocks = 0
    serial_access = 1

@dataclass
class Data:
    name: str = MISSING
    mc:         bool = False
    mode: RandomMode = RandomMode.random_blocks
    seed:        int = -1
    train:       str = ""
    test:        str = ""
    val:         str = ""
    active: Tuple[str] = field(default_factory=list)
    transform1: bool = True
    transform2: bool = True


# @dataclass
# class KryptonIC(Data):
#     name: str = "krypton"
#     path: str = "/data/datasets/NEXT/White-runs/8677/hdf5/prod/v1.2.0/20191122/"
#     trigger: str = "1"
#     # path: str = "/lus/grand/projects/datascience/cadams/datasets/NEXT/new_raw_data/"
#     run: int  = 8677
#     format: dataformat = dataformat.ic

@dataclass
class KryptonMC(Data):
    name: str = "krypton"
    mc:  bool = True
    path: str = "/data/datasets/NEXT/NEW-simulation/Kr/r0/r0_larcv_cuts.r0_merged.h5"


@dataclass
class Krypton8677(Data):
    name: str = "krypton"
    mc:  bool = False
    path: str = "/lus/grand/projects/datascience/cadams/datasets/NEXT/new_raw_data/8677/r8677_krypton_filtered_train.h5"

cs = ConfigStore.instance()
cs.store(group="data", name="krypton_mc", node=KryptonMC)
cs.store(group="data", name="krypton_8677", node=Krypton8677)
