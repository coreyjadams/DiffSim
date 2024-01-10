from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class ModeKind(Enum):
    supervised   = 0
    unsupervised = 1
    iotest       = 2
    inference    = 3
    analysis     = 4

class OptimizerKind(Enum):
    rmsprop  = 0
    adam     = 1
    lamb     = 3
    novograd = 4
    sgd      = 5

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
class TrainBase(Mode):
    checkpoint_iteration:   int           = 500
    summary_iteration:      int           = 1
    logging_iteration:      int           = 1
    weight_decay:           float         = 1e-3

@dataclass
class TrainSupervised(TrainBase):
    optimizer:              OptimizerKind = OptimizerKind.lamb
    loss_power:             float         = 2.0
    name:                   ModeKind      = ModeKind.supervised
    learning_rate:          float         = 0.001
    s2pmt_scaling:          float         = 1.e-5
    s2si_scaling:           float         = 1e0


@dataclass
class TrainUnsupervised(TrainBase):
    optimizer:              OptimizerKind = OptimizerKind.novograd
    name:                   ModeKind      = ModeKind.unsupervised
    learning_rate:          float         = 0.001

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
cs.store(group="mode", name="train_supervised",     node=TrainSupervised)
cs.store(group="mode", name="train_unsupervised",   node=TrainUnsupervised)
cs.store(group="mode", name="iotest", node=IOTest)
cs.store(group="mode", name="inference", node=Inference)
cs.store(group="mode", name="analysis", node=Analysis)
