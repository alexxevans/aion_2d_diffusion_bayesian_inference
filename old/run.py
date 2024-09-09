import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from jax import Array
import os

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


### DATA SETUP ###
def generate_data(key: PRNGKey, n: int):
    key1, key2 = jrandom.split(key, 2)

    # Generate 9 conditioning parameters
    theta = jrandom.normal(key1, (n, 9)) * 3  # Some prior on 9 parameters

    # Generate 5 target parameters
    x = 2 * jnp.sin(theta[:, :5]) + jrandom.normal(key2, (
    n, 5)) * 0.5  # Target variables depend on the first 5 conditioning parameters

    return jnp.concatenate([theta, x], axis=1).reshape(n, -1, 1)  # Now (n, 14, 1)


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


def log_potential(theta: Array, x: Array, sigma_x: float = 0.5, mean_loc: float = 0.0, mean_scale: float = 3.0):
    # Log probability for the 9 conditioning parameters (prior)
    log_prob_theta = jax.scipy.stats.norm.logpdf(theta, mean_loc, mean_scale).sum(axis=-1)

    # Log probability for the 5 target variables
    if x is not None:
        log_prob_x = jax.scipy.stats.norm.logpdf(x, 2 * jnp.sin(theta[:, :5]), sigma_x).sum(axis=-1)
    else:
        log_prob_x = 0

    return log_prob_theta + log_prob_x


def split_data(data, key: PRNGKey, n: int, train_frac: float = 0.8):

    # Split into train and test
    n_train = int(n * train_frac)
    train_data = data[:n_train]
    test_data = data[n_train:]

    # Extract theta (9 parameters) and x (5 targets)
    train_theta, train_x = train_data[:, :9, :], train_data[:, 9:, :]
    test_theta, test_x = test_data[:, :9, :], test_data[:, 9:, :]

    return train_theta, train_x, test_theta, test_x


theta_file = "../data/input/conditioning_data.csv"
x_file = "../data/input/data_to_learn.csv"
data = import_data(jrandom.PRNGKey(1), 849488, theta_file, x_file)

data = data.astype(jnp.float32)  # Convert data to float32

nodes_max = data.shape[1]
node_ids = jnp.arange(nodes_max)

train_theta, train_x, test_theta, test_x = split_data(data, jrandom.PRNGKey(1), 849488)

_ = pairplot(np.array(test_x[..., 0]), labels=["x_1", "x_2", "x_3", "x_4", "x_5"], figsize=(5, 5))
plt.show()



### SETTING UP DIFFUSION PROCESS ###

T = 1.
T_min = 1e-2
sigma_min = 1e-3
sigma_max = 15.

p0 = Independent(Empirical(data), 1) # Empirical distribution of the data
sde = VESDE(p0, sigma_min=sigma_min , sigma_max=sigma_max)

# Scaling fn for the output of the score model
def output_scale_fn(t, x):
    scale = jnp.clip(sde.marginal_stddev(t, jnp.ones_like(x)), 1e-2, None)
    return (1/scale * x).reshape(x.shape)


### BUILDING THE TRANSFORMER ###

dim_value = 20  # Size of the value embedding
dim_id = 20  # Size of the node id embedding
dim_condition = 10  # Size of the condition embedding

def model(t: Array, x: Array, node_ids: Array, condition_mask: Array, edge_mask: Optional[Array] = None):
    """Simplified Simformer model adapted for 9 conditioning parameters and 5 target variables."""
    batch_size, seq_len, _ = x.shape  # Now seq_len should be 14
    condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
    node_ids = node_ids.reshape(-1, seq_len)
    t = t.reshape(-1, 1, 1)

    # Diffusion time embedding
    embedding_time = GaussianFourierEmbedding(64)
    time_embeddings = embedding_time(t)

    # Adjusted embedding sizes for new inputs
    dim_value = 20
    dim_id = 20
    dim_condition = 10

    embedding_net_value = lambda x: jnp.repeat(x, dim_value, axis=-1)
    embedding_net_id = hk.Embed(nodes_max, dim_id, w_init=hk.initializers.RandomNormal(stddev=3.))
    condition_embedding = hk.get_parameter("condition_embedding", shape=(1, 1, dim_condition),
                                           init=hk.initializers.RandomNormal(stddev=0.5))
    condition_embedding = condition_embedding * condition_mask
    condition_embedding = jnp.broadcast_to(condition_embedding, (batch_size, seq_len, dim_condition))

    value_embeddings = embedding_net_value(x)
    id_embeddings = embedding_net_id(node_ids)
    value_embeddings, id_embeddings = jnp.broadcast_arrays(value_embeddings, id_embeddings)

    x_encoded = jnp.concatenate([value_embeddings, id_embeddings, condition_embedding], axis=-1)

    model = Transformer(num_heads=2, num_layers=2, attn_size=10, widening_factor=3)
    h = model(x_encoded, context=time_embeddings, mask=edge_mask)

    out = hk.Linear(1)(h)
    out = output_scale_fn(t, out)
    return out

sample_size = 1024  # A smaller batch size for initialization
sample_data = data[:sample_size]  # Use only a subset of the data

# In Haiku, we need to initialize the model first, before we can use it.
init, model_fn = hk.without_apply_rng(hk.transform(model)) # Init function initializes the parameters of the model, model_fn is the actual model function (which takes the parameters as first argument, hence is a "pure function")
params = init(key, jnp.ones(sample_size), sample_data, node_ids[:sample_size], jnp.zeros_like(node_ids[:sample_size]))

# Here we can see the total number of parameters and their shapes
print("Total number of parameters: ", jax.tree_util.tree_reduce(lambda x,y: x+y, jax.tree_map(lambda x: x.size, params)))
jax.tree_util.tree_map(lambda x: x.shape, params) # Here we can see the shapes of the parameters


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


def loss_fn(params: dict, key: PRNGKey, batch_size: int = 512):
    rng_time, rng_sample, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 5)

    # Generate random times
    times = jax.random.uniform(rng_time, (batch_size, 1, 1), minval=T_min, maxval=1.0)

    # Sample from train data
    idx = jax.random.choice(rng_sample, jnp.arange(train_theta.shape[0]), (batch_size,))
    batch_theta = train_theta[idx]
    batch_x = train_x[idx]
    batch_data = jnp.concatenate([batch_theta, batch_x], axis=1)  # Combine theta and x

    # Other operations remain unchanged
    ids = node_ids
    condition_mask = jax.random.bernoulli(rng_condition, 0.333, shape=(batch_data.shape[0], batch_data.shape[1]))
    condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
    condition_mask *= condition_mask_all_one

    edge_mask = jnp.ones((4 * batch_size // 5, batch_data.shape[1], batch_data.shape[1]), dtype=jnp.bool_)
    marginal_mask = jax.vmap(marginalize, in_axes=(0, None))(jax.random.split(rng_edge_mask1, (batch_size // 5,)),
                                                             edge_mask[0])
    edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
    edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(batch_size,), axis=0)

    loss = denoising_score_matching_loss(params=params, key=rng_sample, times=times, xs_target=batch_data,
                                         loss_mask=condition_mask, model_fn=model_fn,
                                         mean_fn=sde.marginal_mean, std_fn=sde.marginal_stddev,
                                         weight_fn=weight_fn, node_ids=ids, condition_mask=condition_mask,
                                         edge_mask=edge_masks)
    return loss


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
for _ in range(50):
    l = 0
    for i in range(5000):
        key, subkey = jrandom.split(key)
        loss, replicated_params, replicated_opt_state = update(replicated_params, jax.random.split(subkey, (n_devices,)), replicated_opt_state)
        l += loss[0] /5000
    print(l)
params = jax.tree_map(lambda x: x[0], replicated_params)

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

    # Ensure the shape is a tuple
    x_T = jax.random.normal(key1, (shape, len(node_ids))) * end_std[node_ids] + end_mean[node_ids]

    if replace_conditioned:
        x_T = x_T * (1 - condition_mask) + condition_value * condition_mask

    # Solve backward SDE
    keys = jrandom.split(key2, shape)
    ys = jax.vmap(lambda *args: sdeint(*args, noise_type="diagonal"), in_axes=(0, None, None, 0, None), out_axes=0)(
        keys, lambda t, x: drift_backward(t, x, node_ids, condition_mask, edge_mask=edge_mask, score_fn=score_fn,
                                          replace_conditioned=replace_conditioned),
        lambda t, x: diffusion_backward(t, x, node_ids, condition_mask, replace_conditioned=replace_conditioned), x_T,
        jnp.linspace(1., T_min, time_steps))

    return ys


predicted_x = sample_fn(jrandom.PRNGKey(2), test_theta.shape[0], node_ids=jnp.arange(9),  # Pass 9 nodes for theta
                        condition_mask=jnp.zeros((9,), dtype=int), condition_value=jnp.zeros((9,)))

predicted_x = np.array(predicted_x[:, -1, :5])  # Use the last time step and the first 5 dimensions (x_1 to x_5)

true_x = test_x[:, :, 0]  # Extract the target x values (5 outputs)

# Compute the error (e.g., mean squared error)
mse = jnp.mean((predicted_x - true_x) ** 2)  # Only compare the first 5 dimensions (x_1 to x_5)
print(f"Mean Squared Error on test data: {mse}")

# Plot pairwise relationships
with use_style("pyloric"):
    fig, axes = pairplot(predicted_x, figsize=(5, 5), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"], diag_kind="kde")
plt.show()
