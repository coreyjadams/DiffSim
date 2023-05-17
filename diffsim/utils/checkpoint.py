import sys, os
import pathlib

import jax.numpy as numpy

from flax.training import train_state, checkpoints



def save_weights(save_path, model_name, parameters, opt_state, global_step, name = "checkpoint" ):

    # If the file for the model path already exists, we don't change it until after restoring:
    model_path = save_path / pathlib.Path(name) / pathlib.Path("model")
    opt_path   = save_path / pathlib.Path(name) / pathlib.Path("opt")

    # Take the network and snapshot it to file:
    checkpoints.save_checkpoint(
        ckpt_dir = model_path,
        target   = parameters,
        step     = global_step,
        prefix   = model_name,
        keep     = 5)

    checkpoints.save_checkpoint(
        ckpt_dir = opt_path,
        target   = opt_state,
        step     = global_step,
        prefix   = model_name,
        keep     = 5)

def restore_weights(save_path, model_name, name = "checkpoint"):

    # If the file for the model path already exists, we don't change it until after restoring:
    model_path = save_path / pathlib.Path(name) / pathlib.Path("model")
    opt_path   = save_path / pathlib.Path(name) / pathlib.Path("opt")

    # Get the latest checkpoint path:
    latest = checkpoints.latest_checkpoint(model_path, prefix=model_name)

    if latest is None:
        return None, None, None

    global_step =  int(os.path.basename(latest).replace(str(model_name), ""))

    # Take the network and snapshot it to file:
    restored_model = checkpoints.restore_checkpoint(
        ckpt_dir = model_path,
        target   = None,
        prefix   = model_name)

    restored_opt = checkpoints.restore_checkpoint(
        ckpt_dir = opt_path,
        target   = None,
        prefix   = model_name)

    return restored_model, restored_opt, global_step
