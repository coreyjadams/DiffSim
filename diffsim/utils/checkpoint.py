import sys, os
import pathlib

import jax.numpy as numpy

from flax.training import checkpoints

import orbax.checkpoint
from flax.training import orbax_utils
from flax.core import frozen_dict

def init_checkpointer(save_path):

    ckpt_path = save_path / pathlib.Path("checkpoint") / pathlib.Path("model")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_path,
        checkpointer,
        options
    )

    def save_weights(train_state):

        save_args = orbax_utils.save_args_from_target(train_state)

        checkpoint_manager.save(train_state.step, train_state, save_kwargs={'save_args' : save_args})

        return


    def restore_weights(target):

        global_step = checkpoint_manager.latest_step()
        if global_step is None: return None


        checkpoint = checkpoint_manager.restore(global_step, items=target)


        return checkpoint


    return save_weights, restore_weights
