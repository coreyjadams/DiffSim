import sys, os
import jax
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import stax
import jax.tree_util as tree_util # for applying learning rate to gradients


# JAX Is pure functional programming.  So, we're gonna build up to one function here:


# Build the whole thing as a function we can JIT:
# @jit
def simulate_pmts(energy_depositions, parameters, key):
    # Split the key for generation:
    key, subkey = random.split(key)
    generated_electrons, valid_electrons = \
        generate_electrons_batch(energy_depositions, subkey)
    # Split the key for diffusion
    key, subkey = random.split(key)
    diffused = \
        diffuse_electrons_batch(generated_electrons, parameters['diffusion'], subkey)


    lifetime = compute_lifetime(diffused, parameters["lifetime"], valid_electrons)

    # Split off the XY and Z:
    
    diffused_xy = diffused[:,:,:,0:2]
    diffused_z  = diffused[:,:,:,2:]
    
    # Compute the PMT response:
    pmt_response = batch_pmt_nn_apply(parameters["pmt_network"], diffused_xy)**2

    pmt_response = pmt_response * (parameters["pmt_dynamic_range"])**2

    # Compute the waveforms:
    pmt_waveforms = batch_build_waveforms(pmt_response, diffused_z, lifetime, 0.2)
    # Sum over the individual depositions:
    pmt_waveforms = pmt_waveforms.sum(axis=1)
    return pmt_waveforms

# Another critical function is the parameter initialization:
def init_params(key, example_input):

    # The input here needs to be a single electron deposition
    # but we splice that off here, to make it easier.
    
    # Index into the batch dimension and the energy deposit in the batch:
    pmt_input, n_valid = generate_electrons(example_input["energy_deposits"][0][0], key)
    # Diffuse them:
    pmt_input = diffuse_electrons(pmt_input, np.ones(3), key)

    pmt_input_shape = pmt_input[:,0:2].shape
    

    output_size, pmt_network_params = pmt_nn_init(key, pmt_input_shape)

    parameters = {
        "diffusion"   : np.ones(3),
        "lifetime"    : 5000*np.ones(1),
        "pmt_network" : pmt_network_params,
        "pmt_dynamic_range" : np.ones(12),
    }

    return parameters


@jit
def generate_electrons(energy_and_position, key):
    '''
    Generate a sequence of electrons given an energy amount.
    A random number of electrons are generated, but a static-sized array 
    is returned.  This ensures jit'ing is possible later.
    The number of electrons in the array that is valid is also returned
    '''
    
    energy   = energy_and_position[-1]
    position = energy_and_position[0:3]
    
    # For each energy, compute n:
    n = energy* 1000.*1000. / 22.4
    sigmas = np.sqrt(n * 0.15)
    # Generate a sample for each energy:
    n_electrons = (sigmas*random.normal(key) + n).astype(np.int32)
    
    shape = (2000, 3)
    el_arr = np.broadcast_to(position, shape)
    
    return el_arr, n_electrons

# Vmap and jit these out to batches:
generate_electrons_event = jit(vmap(generate_electrons, in_axes=[0, None]))
generate_electrons_batch = jit(vmap(generate_electrons_event, in_axes=[0, None]))


@jit
def diffuse_electrons(electrons, diffusion_scale, key):
    '''
    Apply diffusion to the electrons in a single energy deposition.
    '''
    
    # Then input should be an array of shape [max_electrons, 3]
    # only some of these are valid, but that's ok, it's not
    # that wasteful to diffuse them all.
    
    # Get the z position (we only need this as a scalar!):
    z = electrons[-1,-1]
    n_samples = electrons.shape[0]
    
    # Sample from a normal distribution:
    kicks = random.normal(key, (electrons.shape))
    
    # Scale the kicks by the diffusion scale and sqrt(z)
    return electrons + z*(diffusion_scale**2)*kicks
    

# Likewise, jit and vmap these out:
diffuse_electrons_event = jit(vmap(diffuse_electrons, in_axes=[0,None, None]))
diffuse_electrons_batch = vmap(jit(diffuse_electrons_event), in_axes=[0,None, None])


# The lifetime gets finnicky if I put these two steps into one function.
# I don't know why yet.  But this works.  It also applies 0 weights to invalid electrons.


@jit
def s_compute_probability(_diffused_electrons, _lifetime):
    _z = _diffused_electrons[:,-1]
    probability = np.exp(- _z / _lifetime)

    return probability

@jit
def s_compute_mask(_diffused_electrons, _n_valid):
    # This technique is "stupid" but it jit's and vmap's, so it's not THAT stupid
    arange = np.arange(len(_diffused_electrons))
    mask = arange < _n_valid
    mask = mask.astype("float32")
    return mask

e_compute_probability = jit(vmap(s_compute_probability, in_axes=[0,None]))
compute_probability = jit(vmap(e_compute_probability, in_axes=[0,None]))

e_compute_mask = jit(vmap(s_compute_mask))
compute_mask = jit(vmap(e_compute_mask))
    
@jit
def compute_lifetime(electrons_batch, _lifetime, n_valid_batch):
    return compute_mask(electrons_batch, n_valid_batch) * compute_probability(electrons_batch, _lifetime)


# Define the PMT network and it's worker functions:
pmt_nn_init, pmt_nn_apply = stax.serial(
    stax.Dense(28), stax.Sigmoid,
    stax.Dense(28), stax.Sigmoid,
    stax.Dense(28), stax.Sigmoid,
    stax.Dense(12), stax.LeakyRelu
)


event_pmt_nn_apply = jit(vmap(jit(pmt_nn_apply), in_axes=[None, 0]))
batch_pmt_nn_apply = jit(vmap(event_pmt_nn_apply, in_axes=[None, 0]))


# Functions to build waveforms based on weights and responses:
@jit
def build_waveforms(_sensor_response, _z_positions, _weights, _bin_sigma):
    '''
    Compute the PMT response to electrons on the EL region
    '''
    # This is basically a constant:
    _n_ticks=550
    
    n_electrons = _z_positions.shape[0]
    # Build a range for the exponential input:
    starts = np.zeros(shape=(n_electrons)) + 0.5
    stops  = np.ones(shape=(n_electrons)) * (_n_ticks -1) + 0.5
    
    # Reshape z positions for broadcasting:
    _z_positions = _z_positions.reshape((-1,1))
    
    exp_input = np.linspace(start=starts, stop=stops, num=_n_ticks, axis=-1)

    exp_values = np.exp( - (exp_input - _z_positions)**2.  / (2. * _bin_sigma))
    
    # Normalize the values:
    exp_values = exp_values.transpose() * (0.39894228040/np.sqrt(_bin_sigma))
    
    # Scale by the weights:
    exp_values = exp_values * _weights
    
    waveforms = np.matmul(exp_values, _sensor_response)
    
    return waveforms.transpose()

e_build_waveforms = jit(vmap(build_waveforms, in_axes=[0,0,0,None,]))
batch_build_waveforms = jit(vmap(e_build_waveforms, in_axes=[0,0,0,None,]))




