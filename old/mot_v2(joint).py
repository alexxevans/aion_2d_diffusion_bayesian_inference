import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from jax import Array
import os

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

from functools import partial
from typing import Tuple, List, Optional

import haiku as hk # Neural network library
import optax # Gradient-based optimization in JAX

# Some small helper functions
from probjax.nn.transformers import Transformer

from probjax.nn.helpers import GaussianFourierEmbedding

from library.loss import denoising_score_matching_loss
from probjax.distributions.sde import VESDE
from probjax.distributions import Empirical, Independent

from scoresbibm.utils.plot import use_style

from sbi.analysis import pairplot
import numpy as np

jax.devices() # Should be cuda
_ = os.system("nvidia-smi  --query-gpu=name --format=csv,noheader") # Should show GPU info

key = jax.random.PRNGKey(42)

def generate_data(key: PRNGKey, n: int):
    keys = jrandom.split(key, 11)  # We need 11 keys now: 9 for thetas, 2 for noise
    thetas = jrandom.normal(keys[0], (n, 9)) * 3  # Prior on 9 parameters

    x1 = 2 * jnp.sin(jnp.sum(thetas, axis=1, keepdims=True)) + jrandom.normal(keys[9], (n, 1)) * 0.5
    x2 = 0.1 * jnp.sum(thetas**2, axis=1, keepdims=True) + 0.5 * jnp.abs(x1) * jrandom.normal(keys[10], (n, 1))
    x3 = jnp.cos(thetas[:, 0:1]) + jrandom.normal(keys[1], (n, 1)) * 0.3
    x4 = jnp.exp(thetas[:, 1:2] / 2) + jrandom.normal(keys[2], (n, 1)) * 0.2
    x5 = thetas[:, 2:3] ** 2 + jrandom.normal(keys[3], (n, 1)) * 0.4

    return jnp.concatenate([thetas, x1, x2, x3, x4, x5], axis=1).reshape(n, -1, 1)

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    n = data.shape[0]
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data

def import_data(key: PRNGKey, n: int, theta_file: str, x_file: str):
    # Read the conditioning parameters (theta) from the CSV file
    theta = pd.read_csv(theta_file).values  # Assuming shape (n, 9)

    # Read the target variables (x) from the CSV file
    x = pd.read_csv(x_file).values  # Assuming shape (n, 5)

    # Check the shapes before concatenation
    print(f"Shape of theta: {theta.shape}")
    print(f"Shape of x: {x.shape}")

    # Ensure the number of samples matches n
    if theta.shape[0] != x.shape[0]:
        raise ValueError("Mismatch in number of samples between theta and x")

    # Concatenate theta and x to form the same format as in the original function
    concatenated = jnp.concatenate([theta, x], axis=1)

    # Check shape after concatenation
    print(f"Shape after concatenation: {concatenated.shape}")

    return concatenated.reshape(n, -1, 1)  # Now (n, 14, 1)

theta_file = "../data/input/random/conditioning_data.csv"
x_file = "../data/input/random/data_to_learn.csv"

data = import_data(jrandom.PRNGKey(1), 849488, theta_file, x_file)
data = data.astype(jnp.float32)  # Convert data to float32
nodes_max = data.shape[1]
node_ids = jnp.arange(nodes_max)

train_data, val_data, test_data = split_data(data)

fig, axes = plt.subplots(2, 7, figsize=(28, 8))
labels = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8', 'theta9', 'x1', 'x2', 'x3', 'x4', 'x5']


for i in range(14):
    row = i // 7
    col = i % 7
    axes[row, col].hist(test_data[:, i], bins=50, density=True)
    axes[row, col].set_title(f'Histogram of {labels[i]} GT')
    axes[row, col].set_xlabel(labels[i])
    axes[row, col].set_ylabel('Density')

plt.tight_layout()
plt.show()

### SETTING UP THE DIFFUSION PROCESS ###

# VESDE
T = 1.
T_min = 1e-2
sigma_min = 1e-3
sigma_max = 15.

p0 = Independent(Empirical(train_data), 1)
sde = VESDE(p0, sigma_min=sigma_min , sigma_max=sigma_max)

# Scaling fn for the output of the score model
def output_scale_fn(t, x):
    scale = jnp.clip(sde.marginal_stddev(t, jnp.ones_like(x)), 1e-2, None)
    return (1/scale * x).reshape(x.shape)


### BUILDING THE SIMFORMER ###

dim_value = 20  # Size of the value embedding
dim_id = 20  # Size of the node id embedding
dim_condition = 10  # Size of the condition embedding

def model(t: Array, x: Array, node_ids: Array, condition_mask: Array, edge_mask: Optional[Array] = None):
    """Simplified Simformer model.

    Args:
        t (Array): Diffusion time
        x (Array): Value of the nodes
        node_ids (Array): Id of the nodes
        condition_mask (Array): Condition state of the nodes
        edge_mask (Array, optional): Edge mask. Defaults to None.

    Returns:
        Array: Score estimate of p(x_t)
    """
    batch_size, seq_len, _ = x.shape
    condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
    node_ids = node_ids.reshape(-1, seq_len)
    t = t.reshape(-1, 1, 1)

    # Diffusion time embedding net (here we use a Gaussian Fourier embedding)
    embedding_time = GaussianFourierEmbedding(64)  # Time embedding method
    time_embeddings = embedding_time(t)

    # Tokinization part --------------------------------------------------------------------------------

    embedding_net_value = lambda x: jnp.repeat(x, dim_value,
                                               axis=-1)  # Value embedding net (here we just repeat the value)
    embedding_net_id = hk.Embed(nodes_max, dim_id, w_init=hk.initializers.RandomNormal(
        stddev=3.))  # Node id embedding nets (here we use a learnable random embedding vector)
    condition_embedding = hk.get_parameter("condition_embedding", shape=(1, 1, dim_condition),
                                           init=hk.initializers.RandomNormal(
                                               stddev=0.5))  # Condition embedding (here we use a learnable random embedding vector)
    condition_embedding = condition_embedding * condition_mask  # If condition_mask is 0, then the embedding is 0, otherwise it is the condition_embedding vector
    condition_embedding = jnp.broadcast_to(condition_embedding, (batch_size, seq_len, dim_condition))

    # Embed inputs and broadcast
    value_embeddings = embedding_net_value(x)
    id_embeddings = embedding_net_id(node_ids)
    value_embeddings, id_embeddings = jnp.broadcast_arrays(value_embeddings, id_embeddings)

    # Concatenate embeddings (alternatively you can also add instead of concatenating)
    x_encoded = jnp.concatenate([value_embeddings, id_embeddings, condition_embedding], axis=-1)

    # Transformer part --------------------------------------------------------------------------------
    model = Transformer(num_heads=2, num_layers=2, attn_size=10, widening_factor=3)

    # Encode - here we just use a transformer to transform the tokenized inputs into a latent representation
    h = model(x_encoded, context=time_embeddings, mask=edge_mask)

    # Decode - here we just use a linear layer to get the score estimate (we scale the output by the marginal std dev)
    out = hk.Linear(1)(h)
    out = output_scale_fn(t, out)  # SDE dependent output scaling
    return out

# Choose a small batch size for initialization
init_batch_size = 128

# Create a small batch of data for initialization
init_data = jax.tree_map(lambda x: x[:init_batch_size], data)
init_node_ids = node_ids[:init_batch_size]

# Initialize the model with the small batch
init, model_fn = hk.without_apply_rng(hk.transform(model))
params = init(key, jnp.ones(init_batch_size), init_data, init_node_ids, jnp.zeros_like(init_node_ids))

# Here we can see the total number of parameters and their shapes
print("Total number of parameters: ", jax.tree_util.tree_reduce(lambda x,y: x+y, jax.tree_map(lambda x: x.size, params)))
jax.tree_util.tree_map(lambda x: x.shape, params) # Here we can see the shapes of the parameters


### LOSS ###

def weight_fn(t: Array):
    # MLE weighting
    return jnp.clip(sde.diffusion(t, jnp.ones((1, 1, 1))) ** 2, 1e-4)


def marginalize(rng: PRNGKey, edge_mask: Array):
    # Simple function that marginializes out a single node from a adjacency matrix of a graph.
    idx = jax.random.choice(rng, jnp.arange(edge_mask.shape[0]), shape=(1,), replace=False)
    edge_mask = edge_mask.at[idx, :].set(False)
    edge_mask = edge_mask.at[:, idx].set(False)
    edge_mask = edge_mask.at[idx, idx].set(True)
    return edge_mask


def loss_fn(params: dict, key: PRNGKey, batch_size: int = 1024):
    rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)

    # Generate data and random times
    times = jax.random.uniform(rng_time, (batch_size, 1, 1), minval=T_min, maxval=1.0)
    batch_xs = generate_data(rng_data, batch_size)  # n, T_max, 1

    # Node ids (can be subsampled but here we use all nodes)
    ids = node_ids

    # Condition mask -> randomly condition on some data.
    condition_mask = jax.random.bernoulli(rng_condition, 0.333, shape=(batch_xs.shape[0], batch_xs.shape[1]))
    condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
    condition_mask *= condition_mask_all_one  # Avoid conditioning on all nodes -> nothing to train...
    condition_mask = condition_mask[..., None]
    # Alternatively you can also set the condition mask manually to specific conditional distributions.
    # condition_mask = jnp.zeros((3,), dtype=jnp.bool_)  # Joint mask
    # condition_mask = jnp.array([False, True, True], dtype=jnp.bool_)  # Posterior mask

    #condition_mask = jnp.array(
    #    [True, True, True, True, True, True, True, True, True, False, False, False, False, False],
    #    dtype=jnp.bool_)  # Likelihood mask
    #condition_mask = jnp.tile(condition_mask[None, :, None], (batch_size, 1, 1))

    # You can also structure the base mask!
    edge_mask = jnp.ones((4 * batch_size // 5, batch_xs.shape[1], batch_xs.shape[1]),
                         dtype=jnp.bool_)  # Dense default mask

    # Optional: Include marginal consistency
    marginal_mask = jax.vmap(marginalize, in_axes=(0, None))(jax.random.split(rng_edge_mask1, (batch_size // 5,)),
                                                             edge_mask[0])
    edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
    edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(batch_size,),
                                   axis=0)  # Randomly choose between dense and marginal mask

    # Forward diffusion, do not perturb conditioned data
    # Will use the condition mask to mask to prevent adding noise for nodes that are conditioned.
    loss = denoising_score_matching_loss(params, rng_sample, times, batch_xs, condition_mask, model_fn=model_fn,
                                         mean_fn=sde.marginal_mean, std_fn=sde.marginal_stddev, weight_fn=weight_fn,
                                         node_ids=ids, condition_mask=condition_mask, edge_mask=edge_masks)

    return loss

def calculate_validation_loss(params, val_data, batch_size=1024):
    num_batches = len(val_data) // batch_size
    total_loss = 0
    for i in range(num_batches):
        batch = val_data[i*batch_size:(i+1)*batch_size]
        key = jrandom.PRNGKey(i)  # Use a different key for each batch
        loss = loss_fn(params, key, batch_size)
        total_loss += loss
    return total_loss / num_batches


### TRAINING ###

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@partial(jax.pmap, axis_name="num_devices")
def update(params, rng, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng)

    loss = jax.lax.pmean(loss, axis_name="num_devices")
    grads = jax.lax.pmean(grads, axis_name="num_devices")

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
replicated_opt_state = jax.tree_map(lambda x: jnp.array([x] * n_devices), opt_state)

key = jrandom.PRNGKey(0)
num_epochs = 5
steps_per_epoch = 5000

for epoch in range(num_epochs):
    train_loss = 0
    for i in range(steps_per_epoch):
        key, subkey = jrandom.split(key)
        loss, replicated_params, replicated_opt_state = update(replicated_params,
                                                               jax.random.split(subkey, (n_devices,)),
                                                               replicated_opt_state)
        train_loss += loss[0] / steps_per_epoch

    # Calculate validation loss
    params = jax.tree_map(lambda x: x[0], replicated_params)
    val_loss = calculate_validation_loss(params, val_data)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

params = jax.tree_map(lambda x: x[0], replicated_params)


### SAMPLING ###

from functools import partial
from probjax.utils.sdeint import sdeint

condition_mask = jnp.zeros((nodes_max,))
condition_value = jnp.zeros((nodes_max,))

# Reverse SDE drift
def drift_backward(t, x, node_ids=node_ids, condition_mask=condition_mask, edge_mask=None, score_fn=model_fn,
                   replace_conditioned=True):
    score = score_fn(params, t.reshape(-1, 1, 1), x.reshape(-1, len(node_ids), 1), node_ids,
                     condition_mask[:len(node_ids)], edge_mask=edge_mask)
    score = score.reshape(x.shape)

    f = sde.drift(t, x) - sde.diffusion(t, x) ** 2 * score
    if replace_conditioned:
        f = f * (1 - condition_mask[:len(node_ids)])

    return f

# Reverse SDE diffusion
def diffusion_backward(t, x, node_ids=node_ids, condition_mask=condition_mask, replace_conditioned=True):
    b = sde.diffusion(t, x)
    if replace_conditioned:
        b = b * (1 - condition_mask[:len(node_ids)])
    return b


end_std = jnp.squeeze(sde.marginal_stddev(jnp.ones(1)))
end_mean = jnp.squeeze(sde.marginal_mean(jnp.ones(1)))

@partial(jax.jit, static_argnums=(1, 3, 7, 8))
def sample_fn(key, shape, node_ids=node_ids, time_steps=500, condition_mask=jnp.zeros((nodes_max,), dtype=int),
              condition_value=jnp.zeros((nodes_max,)), edge_mask=None, score_fn=model_fn, replace_conditioned=True):
    condition_mask = condition_mask[:len(node_ids)]
    key1, key2 = jrandom.split(key, 2)
    # Sample from noise distribution at time 1
    x_T = jax.random.normal(key1, shape + (len(node_ids),)) * end_std[node_ids] + end_mean[node_ids]

    if replace_conditioned:
        x_T = x_T * (1 - condition_mask) + condition_value * condition_mask
    # Solve backward sde
    keys = jrandom.split(key2, shape)
    ys = jax.vmap(lambda *args: sdeint(*args, noise_type="diagonal"), in_axes=(0, None, None, 0, None), out_axes=0)(
        keys, lambda t, x: drift_backward(t, x, node_ids, condition_mask, edge_mask=edge_mask, score_fn=score_fn,
                                          replace_conditioned=replace_conditioned),
        lambda t, x: diffusion_backward(t, x, node_ids, condition_mask, replace_conditioned=replace_conditioned), x_T,
        jnp.linspace(1., T_min, time_steps))
    return ys


def batch_sample(key, thetas, batch_size=1000):
    num_samples = thetas.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    samples_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_thetas = thetas[start_idx:end_idx]
        batch_condition_value = jnp.pad(batch_thetas, ((0, 0), (0, 5)), mode='constant')

        key, subkey = jax.random.split(key)
        batch_samples = sample_fn(subkey, (batch_thetas.shape[0],), node_ids,
                                  condition_mask=condition_mask,
                                  condition_value=batch_condition_value)

        samples_list.append(batch_samples[:, -1, :])  # Keep only the final timestep

    return jnp.concatenate(samples_list, axis=0)

# Assuming test_data is already split
test_thetas = test_data[:, :9, 0]  # This will have shape (n_test, 9)
print("Shape of test_thetas:", test_thetas.shape)

# Set up conditioning mask and values for likelihood sampling
condition_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # Condition on 9 thetas, sample 5 xs
condition_value = jnp.pad(test_thetas, ((0, 0), (0, 5)), mode='constant')  # Pad 9 thetas with zeros for 5 xs
print("Shape of condition_value:", condition_value.shape)

# Sample from the likelihood using the test thetas
key_test = jrandom.PRNGKey(42)  # New key for test sampling
final_samples = batch_sample(key_test, test_thetas)

print("Shape of final_samples:", final_samples.shape)

fig, axes = plt.subplots(2, 7, figsize=(28, 8))
labels = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8', 'theta9', 'x1', 'x2', 'x3', 'x4', 'x5']

for i in range(14):
    row = i // 7
    col = i % 7
    axes[row, col].hist(final_samples[:, i], bins=50, density=True)
    axes[row, col].set_title(f'Histogram of {labels[i]} Predictions')
    axes[row, col].set_xlabel(labels[i])
    axes[row, col].set_ylabel('Density')

plt.tight_layout()
plt.show()