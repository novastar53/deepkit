import copy
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx


from .utils import global_l2_norm, cosine


def zero_out_next_layers(layer_prefix, params):
    state = {"prev": True}

    def mask_fn(path, leaf):
        if state["prev"] is False:
            return jnp.zeros_like(leaf)
        path = ".".join(
            [str(p.key) for p in path if isinstance(p, jax.tree_util.DictKey)]
        )
        if layer_prefix in path:
            state["prev"] = False
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(mask_fn, params)


def zero_out_OptVariables(v):
    zeros = jnp.zeros_like(v)
    return nnx.training.optimizer.OptVariable(
        source_type=nnx.variablelib.Param, value=zeros
    )


def zero_out_next_optimizer_states(params, conv_layer_id):
    state = {"prev": True}
    layer_prefix = f"convs.{conv_layer_id}"

    def mask_fn(path, leaf):
        if state["prev"] is False:
            return zero_out_OptVariables(leaf)
        path = ".".join(
            [str(p.key) for p in path if isinstance(p, jax.tree_util.DictKey)]
        )
        if layer_prefix in path:
            state["prev"] = False
            return zero_out_OptVariables(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(mask_fn, params)


def loss_fn(model, batch, targets):
    logits, activations = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, activations


@nnx.jit(static_argnums=[4])
def calc_ics(optimizer, batch, labels, grads, wrt_layer_id):
    """Calculate the gradient-based internal covariate as per 
       Santurkar et al. (2018)."""

    wrt_layer = f"convs.{wrt_layer_id}"
    # Set the velocities of the next layers to zero to prevent them from updating
    # the weights
    masked_opt_state = zero_out_next_optimizer_states(optimizer.opt_state, wrt_layer_id)
    optimizer.opt_state = masked_opt_state

    # zero out the gradients of the next layers
    masked_grads = zero_out_next_layers(wrt_layer, grads)

    # update the weights of the previous layers
    optimizer.update(masked_grads)

    # calculate the lookahead gradients
    (_, _), lookahead_grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        optimizer.model, batch, labels
    )

    # pick out the gradient wrt current layer
    wrt_layer_grad = grads["convs"][wrt_layer_id]
    lookahead_grad = lookahead_grads["convs"][wrt_layer_id]

    # calculate the ICS measures
    l2_diff = jax.tree_util.tree_map(
        lambda x, y: jnp.linalg.norm(x.flatten() - y.flatten()),
        wrt_layer_grad,
        lookahead_grad,
    )
    cos_angle = jax.tree_util.tree_map(cosine, wrt_layer_grad, lookahead_grad)

    return l2_diff, cos_angle


def santurkar_ics_step(
    optimizer: nnx.Optimizer, grads, batch: jax.Array, labels: jax.Array
):
    """Calculate the gradient-based ICS for all the convolutional layers"""

    ics_results = []
    for i in range(len(optimizer.model.convs)):
        optimizer_copy = optimizer.__deepcopy__()
        ics_measures = calc_ics(optimizer_copy, batch, labels, grads, i)
        ics_results.append(ics_measures)

    return ics_results


@nnx.jit(static_argnums=[4, 5, 6, 7])
def loss_landscape_step(
    model, batch, targets, grads, lr: float, min_step=0.5, max_step=4.2, step_size=0.3
):
    """Approximate the Lipschitzness of the loss landscape as per 
       Santurkar et al. (2018)."""

    def calc_loss(model, grads, lr, s):
        graphdef, state, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
        updated_state = jax.tree_util.tree_map(
            lambda x, y: x - lr * s * y, state, grads
        )
        model = nnx.merge(graphdef, updated_state, batch_stats)
        logits, _ = model(batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss

    _calc = partial(calc_loss, model, grads, lr)
    scales = jnp.arange(min_step, max_step, step_size)
    _vmap_calc = nnx.vmap(_calc, in_axes=0)
    losses = _vmap_calc(scales)
    min_loss = jnp.min(losses)
    max_loss = jnp.max(losses)
    return min_loss, max_loss


@nnx.jit(static_argnums=[4, 5, 6, 7])
def grad_landscape_step(
    model, batch, targets, grads, lr: float, min_step=0.5, max_step=4.2, step_size=0.3
):
    """Approximate the Lipschitzness of the gradient norm landscape as per
       Santurkar et al. (2018)"""

    def calc_norm(model, grads, lr, s):

        def loss_fn(model, batch, targets):
            logits, _ = model(batch)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()
            return loss

        graphdef, state, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
        updated_state = jax.tree_util.tree_map(
            lambda x, y: x - lr * s * y, state, grads
        )
        model = nnx.merge(graphdef, updated_state, batch_stats)
        new_grads = nnx.grad(loss_fn)(model, batch, targets)
        grad_diff = jax.tree_util.tree_map(lambda x, y: x - y, new_grads, grads)
        l2 = global_l2_norm(grad_diff)
        return l2

    _grad_fn = partial(calc_norm, model, grads, lr)
    scales = jnp.arange(min_step, max_step, step_size)
    _vmap_grad_fn = nnx.vmap(_grad_fn, in_axes=0)
    grad_diffs = _vmap_grad_fn(scales)
    min_diff = jnp.min(grad_diffs)
    max_diff = jnp.max(grad_diffs)
    return min_diff, max_diff
