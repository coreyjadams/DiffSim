import jax.numpy as numpy
import jax.random as random

from jax import tree_util, vmap
from functools import reduce


def init_rng_keys(key, key_list):
    all_leaves, tree_def = tree_util.tree_flatten(key_list)
    n_leaves = len(all_leaves)

    new_keys = numpy.split(random.split(key, n_leaves).reshape((-1)), n_leaves)
    new_keys = [{ key : value} for key, value in zip(all_leaves, new_keys)]
    return reduce(lambda x, y : {**x, **y}, new_keys)


def update_rng_keys(key, key_list):
    all_leaves, tree_def = tree_util.tree_flatten(key_list)
    n_leaves = len(all_leaves)

    new_keys = numpy.split(random.split(key, n_leaves).reshape((-1,)), n_leaves)
    # new_keys = [ { key : value} for key, value in zip(all_leaves, new_keys) ]

    new_key_tree = tree_util.tree_unflatten(tree_def, new_keys)

    return new_key_tree

def batch_update_rng_keys(key_list):

    f = lambda x : vmap(random.split, in_axes=(0,None))(x, 1).reshape((-1,2))

    keys = tree_util.tree_map(f, key_list)

    return keys

def batch_init_rng_keys(key_list, batch_size):

    split_keys = tree_util.tree_map(
        lambda x : random.split(x, batch_size),
        key_list
    )

    return split_keys


from . NEW_Simulator_flax import init_NEW_simulator

def init_simulator(init_key, config, example_data):

    # print(example_data["e_deps"].shape)
    import flax

    from jax import vmap, jit

    # Initialize the simulator object, which has a number of 
    # hooks for jax RNGs and we need to see them too
    simulator, rng_key_names = init_NEW_simulator(config.physics)



    # Split the input key:
    init_key, subkey = random.split(init_key)

    # Initialize the model's keys:
    rng_keys = init_rng_keys(subkey, rng_key_names)

    # Split for the next keys:
    init_key, subkey = random.split(init_key)
    next_rng_keys = update_rng_keys(subkey, rng_keys)
    

    # Add a key for the weight initialization too:
    init_key, subkey = random.split(init_key)
    rng_keys.update({"params" : subkey})

    # Vectorize the module:

    leaves, tree_def = tree_util.tree_flatten(rng_keys)
    leaves = [ True for leaf in leaves ]
    split_rng = tree_util.tree_unflatten(tree_def, leaves)
    split_rng["params"] = False

    # Initialize the parameters:
    sim_params = simulator.init(rng_keys, example_data['e_deps'][0])
    sim_func   = jit(flax.linen.apply(type(simulator).__call__, simulator))

    simulator_str = simulator.tabulate(rng_keys, example_data["e_deps"][0],
        console_kwargs={"width":120}, depth=3)

    # print(example_data.keys(), flush=True)
    # print(example_data["S2Si"].shape, flush=True)
    # exit()

    batch_size = example_data["S2Si"].shape[0]
    multi_rngs = batch_init_rng_keys(rng_keys, batch_size)
    # print(multi_rngs, flush=True)
    sim_func = jit(vmap(sim_func, in_axes=(None, 0,)))
    
    # test_output = sim_func(sim_params, example_data['e_deps'], rngs=multi_rngs)

    return sim_func, sim_params, multi_rngs, simulator_str