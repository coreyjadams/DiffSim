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

    def save_weights(train_state, disc_state):

        if disc_state is None:
            save_args = orbax_utils.save_args_from_target(train_state)
            print(save_args)
        else:
            save_args = orbax_utils.save_args_from_target([train_state, disc_state])
            print(save_args)

        checkpoint_manager.save(train_state.step, train_state, save_kwargs={'save_args' : save_args})

        return


    def restore_weights(target, disc_target):

        global_step = checkpoint_manager.latest_step()
        if global_step is None: return None, None


        if disc_target is not None:
            target = [target, disc_target]

        checkpoint = checkpoint_manager.restore(global_step, items=target)
        print(checkpoint)
        if disc_target is None or checkpoint is None:

            return checkpoint, None
        else:
            return checkpoint[0], checkpoint[1]


    return save_weights, restore_weights
