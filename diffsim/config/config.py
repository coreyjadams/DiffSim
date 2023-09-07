from enum import Enum

from dataclasses import dataclass, field
from typing import List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
cs = ConfigStore.instance()

# from .network   import Network
from .mode      import *
from .data      import Data
from .physics   import Simulator

class ComputeMode(Enum):
    CPU   = 0
    GPU   = 1
    DPCPP = 2

class Precision(Enum):
    float32  = 0
    mixed    = 1
    bfloat16 = 2
    float16  = 3

@dataclass
class Run:
    distributed:        bool        = True
    compute_mode:       ComputeMode = ComputeMode.GPU
    iterations:         int         = 500
    minibatch_size:     int         = 32
    id:                 str         = MISSING
    precision:          Precision   = Precision.float32
    profile:            bool        = False
    checkpoint:         int         = 200
    image_iteration:    int         = 200

cs.store(group="run", name="base_run", node=Run)


cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": True, "root": {"handlers": []}},
)


defaults = [
    {"run" : "base_run"},
    # {"mode": "train_unsupervised"},
    {"data": "krypton"},
    {"physics":"NEW"},
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    #         "_self_",
    #         {"run" : Run()}
    #     ]
    # )

    physics: Simulator = MISSING
    run:         Run = MISSING
    mode:       Mode = MISSING
    data:       Data = MISSING
    seed:        int = 0 
    save_path:   str = MISSING
    model_name:  str = MISSING

cs.store(name="base_config", node=Config)
