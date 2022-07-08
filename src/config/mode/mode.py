from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class ModeKind(Enum):
    train     = 0
    iotest    = 1
    inference = 2
    analysis  = 3

class OptimizerKind(Enum):
    SGD = 0
    Adam = 1

class Loss(Enum):
    MSE = 0
    MAE = 1
    ED  = 2

@dataclass
class Mode:
    name:               ModeKind = MISSING
    no_summary_images:  bool     = False
    weights_location:   str      = ""

@dataclass
class Train(Mode):
    checkpoint_iteration:   int           = 500
    summary_iteration:      int           = 1
    logging_iteration:      int           = 1
    optimizer:              OptimizerKind = OptimizerKind.Adam
    loss:                   Loss          = Loss.MSE
    name:                   ModeKind      = ModeKind.train
    learning_rate:          float         = 0.1

@dataclass
class Inference(Mode):
    start_index:        int      = 0
    summary_iteration:  int      = 1
    logging_iteration:  int      = 1
    name:               ModeKind = ModeKind.inference

@dataclass
class Analysis(Mode):
    name:               ModeKind = ModeKind.analysis

@dataclass
class IOTest(Mode):
    name:               ModeKind = ModeKind.iotest


cs = ConfigStore.instance()
cs.store(group="mode", name="train", node=Train)
cs.store(group="mode", name="iotest", node=IOTest)
cs.store(group="mode", name="inference", node=Inference)
cs.store(group="mode", name="analysis", node=Analysis)
