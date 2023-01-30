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
    path: str = "/data/datasets/NEXT/White-runs/8677/hdf5/prod/v1.2.0/20191122/"
    # path: str = "/lus/grand/projects/datascience/cadams/datasets/NEXT/new_raw_data/"
    run: int  = 8677


cs = ConfigStore.instance()
cs.store(group="data", name="krypton", node=Krypton)
