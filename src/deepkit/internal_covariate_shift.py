import copy
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

def calc_l2_norm(x: nnx.statelib.State):
    leaves, _ = jax.tree_util.tree_flatten(x)
    squared_sum = sum([jnp.sum(jnp.square(l)) for l in leaves])
    return jnp.sqrt(squared_sum)


def cosine(x: jnp.ndarray, y: jnp.ndarray):
    x = x.flatten()
    y = y.flatten()
    k = jnp.dot(x, y)
    return k /(jnp.linalg.norm(x)*jnp.linalg.norm(y))

def zero_out_next_layers(layer_prefix, params):
    state = {"prev": True}
    def mask_fn(path, leaf):
        if state["prev"] is False: 
            return jnp.zeros_like(leaf)
        path = ".".join([str(p.key) for p in path if isinstance(p, jax.tree_util.DictKey)])
        if layer_prefix in path:
            state["prev"] = False 
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(mask_fn, params)

def zero_out_OptVariables(v):
    zeros = jnp.zeros_like(v)
    return nnx.training.optimizer.OptVariable(source_type=nnx.variablelib.Param, value=zeros)

def zero_out_next_optimizer_states(params, conv_layer_id):
    state = {"prev": True}
    layer_prefix = f"convs.{conv_layer_id}"
    def mask_fn(path, leaf):
        if state["prev"] is False: 
            return zero_out_OptVariables(leaf)
        path = ".".join([str(p.key) for p in path if isinstance(p, jax.tree_util.DictKey)])
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

    wrt_layer = f"convs.{wrt_layer_id}"
    # print("layer id", wrt_layer)
    #nnx.display(opt_state)
    # print("grad [conv 0]", grads.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"grad [conv {wrt_layer_id}]", grads.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"grad [conv 9]", grads.convs[9].conv.kernel.value[0,0,0,0])
    # Set the velocities of the next layers to zero to prevent them from updating
    # the weights 
    # print("opt state [conv 0]", optimizer.opt_state[0].trace.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"opt state [conv {wrt_layer_id}]", optimizer.opt_state[0].trace.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"opt state [conv 9]", optimizer.opt_state[0].trace.convs[9].conv.kernel.value[0,0,0,0])
    masked_opt_state = zero_out_next_optimizer_states(optimizer.opt_state, wrt_layer_id)
    optimizer.opt_state = masked_opt_state
    # print("masked opt state [conv 0]", masked_opt_state[0].trace.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"masked opt state [conv {wrt_layer_id}]", masked_opt_state[0].trace.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"masked opt state [conv 9]", masked_opt_state[0].trace.convs[9].conv.kernel.value[0,0,0,0])

    # zero out the gradients of the next layers
    masked_grads = zero_out_next_layers(wrt_layer, grads)
    # print("masked grad [conv 0]", masked_grads.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"masked grad [conv {wrt_layer_id}]", masked_grads.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"masked grad [conv 9]", masked_grads.convs[9].conv.kernel.value[0,0,0,0])

    # update the weights of the previous layers
    # print("weight [conv 0]", optimizer.model.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"weight [conv {wrt_layer_id}]", optimizer.model.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"weight [conv 9]", optimizer.model.convs[9].conv.kernel.value[0,0,0,0])
    optimizer.update(masked_grads)
    # print("updated weight [conv 0]", optimizer.model.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"updated weight [conv {wrt_layer_id}]", optimizer.model.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"updated weight [conv 9]", optimizer.model.convs[9].conv.kernel.value[0,0,0,0])

    # calculate the lookahead gradients
    (_, _), lookahead_grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model, batch, labels)
    ## print("lookahead loss", lookahead_loss)
    # print("lookahead grad [conv 0]", lookahead_grads.convs[0].conv.kernel.value[0,0,0,0])
    # print(f"lookahead grad [conv {wrt_layer_id}]", lookahead_grads.convs[wrt_layer_id].conv.kernel.value[0,0,0,0])
    # print(f"lookahead grad [conv 9]", lookahead_grads.convs[9].conv.kernel.value[0,0,0,0])

    # pick out the gradient wrt current layer
    wrt_layer_grad = grads["convs"][wrt_layer_id] 
    lookahead_grad = lookahead_grads["convs"][wrt_layer_id]

    # calculate the ICS measures
    l2_diff = jax.tree_util.tree_map(lambda x, y: jnp.linalg.norm(x.flatten() - y.flatten()), wrt_layer_grad, lookahead_grad)
    cos_angle = jax.tree_util.tree_map(cosine, wrt_layer_grad, lookahead_grad)

    ## print("l2_norm [conv 3]", l2_norm.conv.kernel.value)
    ## print("cosine angle [conv 3]", cos_angle.conv.kernel.value)
    return l2_diff, cos_angle
 

def santurkar_ics_step(optimizer: nnx.Optimizer, 
                       grads,
                       batch: jax.Array, 
                       labels: jax.Array):

    ics_results = []
    for i in range(len(optimizer.model.convs)):
        optimizer_copy = optimizer.__deepcopy__()
        ics_measures = calc_ics(optimizer_copy, batch, labels, grads, i)
        ics_results.append(ics_measures)
    
    return ics_results


def loss_landscape_step(model, batch, targets, lr: float, step_size=0.3):

    @nnx.jit
    def calc_loss(model, grads, lr, s):

        def loss_fn(model, batch, targets):
            logits, _ = model(batch)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss

        graphdef, state, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
        updated_state = jax.tree_util.tree_map(lambda x, y: x - lr*s*y, state, grads)
        model = nnx.merge(graphdef, updated_state, batch_stats)
        loss, _ = nnx.value_and_grad(loss_fn)(model, batch, targets)
        return loss

    min_loss, max_loss = float('inf'), 0
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, targets)
    min_loss = min(loss, min_loss)
    max_loss = max(loss, max_loss)
    scales = jnp.arange(0.5, 4.0+step_size, step_size)
    for s in scales:
        loss = calc_loss(model, grads, lr, s)
        min_loss = min(loss, min_loss)
        max_loss = max(loss, max_loss)
    return min_loss, max_loss

def grad_landscape_step(model, batch, targets, lr: float):

    def loss_fn(model, batch, targets):
        logits, _ = model(batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss

    @nnx.jit
    def calc_norm(model, grads, lr, s):
        graphdef, state, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
        updated_state = jax.tree_util.tree_map(lambda x, y: x - lr*s*y, state, grads)
        model = nnx.merge(graphdef, updated_state, batch_stats)
        __, grads = nnx.value_and_grad(loss_fn)(model, batch, targets)
        norm = calc_l2_norm(grads)
        return norm

    min_norm, max_norm = float('inf'), 0
    _, grads = nnx.value_and_grad(loss_fn)(model, batch, targets)
    scales = jnp.arange(0.5, 4.2, 0.3)
    grad_norm = calc_l2_norm(grads)
    min_norm = min(grad_norm, min_norm)
    max_norm = max(grad_norm, max_norm)
    for s in scales:
        grad_norm = calc_norm(model, grads, lr, s)
        min_norm = min(grad_norm, min_norm)
        max_norm = max(grad_norm, max_norm)
    return min_norm, max_norm

