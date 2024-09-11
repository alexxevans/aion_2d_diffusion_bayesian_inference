
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from functools import partial

from typing import Optional, Sequence, Union, Callable
from jaxtyping import PyTree, Array

def denoising_score_matching_loss(
        params: PyTree,
        key: PRNGKey,
        times: Array,
        xs_target: Array,
        loss_mask: Optional[Array],
        *args,
        model_fn: Callable,
        mean_fn: Callable,
        std_fn: Callable,
        weight_fn: Callable,
        axis: int = -2,
        rebalance_loss: bool = False,
        **kwargs,
) -> Array:
    """This function computes the denoising score matching loss. Which can be used to train diffusion models.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random generator key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        loss_mask (Optional[Array]): Mask for the target distribution. If None, no mask is applied, should be broadcastable to shape (batch_size, 1).
        model_fn (Callable): Score model that takes parameters, times, and samples as input and returns the score. Should be a function of the form model_fn(params, times, xs_t, *args) -> s_t.
        mean_fn (Callable): Mean function of the SDE.
        std_fn (Callable): Std function of the SDE.
        weight_fn (Callable): Weight function for the loss.
        axis (int, optional): Axis to sum over. Defaults to -2.


    Returns:
        Array: Loss
    """
    eps = jax.random.normal(key, shape=xs_target.shape)
    mean_t = mean_fn(times, xs_target)
    std_t = std_fn(times, xs_target)
    xs_t = mean_t + std_t * eps

    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)

    score_pred = model_fn(params, times, xs_t, *args, **kwargs)
    score_target = -eps / std_t

    base_loss = (score_pred - score_target) ** 2
    if loss_mask is not None:
        base_loss = jnp.where(loss_mask, 0.0,base_loss)
    base_loss = weight_fn(times) * jnp.sum(base_loss, axis=axis, keepdims=True)
    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
        base_loss = jnp.where(num_elements > 0, base_loss / num_elements, 0.0)
    base_loss = jnp.mean(base_loss)

    ### 2ND ORDER ADJUSTMENTS ###
    score_hess=0
    #key, subkey = jax.random.split(key)
    #hutchinson_eps = jax.random.normal(subkey, shape=xs_t.shape)  # {epsilon} generates noise

    # model predicted
    #score_pred_noisy = model_fn(params, times, xs_t + hutchinson_eps, *args, **kwargs)  # {s_phi(perturbed data)}
    #score_hess = jnp.mean(jnp.sum(hutchinson_eps * (score_pred_noisy - score_pred), axis=-1))  # {hessian trace}

    gamma = 1e-3

    reg_term = gamma * score_hess
    loss = base_loss + reg_term

    return loss