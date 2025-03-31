import copy

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx


def cosine(x: jnp.ndarray, y: jnp.ndarray):
    x = x.flatten()
    y = y.flatten()
    k = jnp.dot(x, y)
    return k /(jnp.linalg.norm(x)*jnp.linalg.norm(y))

def zero_out_prev_layers(layer_prefix, params):
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

def zero_out_prev_optimizer_states(layer_prefix, params):
    state = {"prev": True}
    def mask_fn(path, leaf):
        if state["prev"] is False: 
            return zero_out_OptVariables(leaf)
        path = ".".join([str(p.key) for p in path if isinstance(p, jax.tree_util.DictKey)])
        if layer_prefix in path:
            state["prev"] = False 
            return zero_out_OptVariables(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(mask_fn, params)

@nnx.jit(static_argnums=[4,])
def calc_ics(model: nnx.Module, optimizer: nnx.Optimizer, batch: jax.Array, labels: jax.Array, wrt_layer: str):

    def loss_fn(model, batch, targets):
        logits, activations = model(batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss, activations
    
    # create copies of the model and optimizer 
    optimizer_copy = optimizer.__deepcopy__()
    #print("weight [conv 0]", optimizer_copy.model.convs[0].conv.kernel.value[0,0,0,0])
    #print("weight [conv 3]", optimizer_copy.model.convs[3].conv.kernel.value[0,0,0,0])
    # calculate the gradient
    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer_copy.model, batch, labels)
    #print("loss", loss)
    #print("grad [conv 0]", grads.convs[0].conv.kernel.value[0,0,0,0])
    #print("grad [conv 3]", grads.convs[3].conv.kernel.value[0,0,0,0])
    # zero out the optimizer velocities of the previous layers
    opt_state = optimizer_copy.opt_state
    opt_state = zero_out_prev_optimizer_states(wrt_layer, opt_state)
    #nnx.display(opt_state)
    optimizer_copy.opt_state = opt_state
    #nnx.display(opt_state)
    #print("param [conv 0]", optimizer_copy.model.convs[0].conv.kernel.value[0,0,0,0])
    #print("param [conv 3]", optimizer_copy.model.convs[3].conv.kernel.value[0,0,0,0])
    #print("opt state [conv 0]", optimizer_copy.opt_state[0].trace.convs[0].conv.kernel.value[0,0,0,0])
    #print("opt state [conv 3]", optimizer_copy.opt_state[0].trace.convs[3].conv.kernel.value[0,0,0,0])
    # zero out the gradients of the previous layers
    prev_grads = zero_out_prev_layers(wrt_layer, grads)
    #print("prev grad [conv 0]", prev_grads.convs[0].conv.kernel.value[0,0,0,0])
    #print("prev grad [conv 3]", prev_grads.convs[3].conv.kernel.value[0,0,0,0])
    # update the weights of the previous layers
    optimizer_copy.update(prev_grads)
    #print("updated weight [conv 0]", optimizer_copy.model.convs[0].conv.kernel.value[0,0,0,0])
    #print("updated weight [conv 3]", optimizer_copy.model.convs[3].conv.kernel.value[0,0,0,0])
    # calculate the gradient
    # calculate the lookahead loss and gradients
    (lookahead_loss, _), lookahead_grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer_copy.model, batch, labels)
    #print("lookahead loss", lookahead_loss)
    #print("shifted grad [conv 0]", lookahead_grads.convs[0].conv.kernel.value[0,0,0,0])
    #print("shifted grad [conv 3]", lookahead_grads.convs[3].conv.kernel.value[0,0,0,0])
    # pick out the gradient wrt current layer
    conv_layer_id = int(wrt_layer.split(".")[1])
    wrt_layer_grad = grads["convs"][conv_layer_id] 
    shifted_grad = lookahead_grads["convs"][conv_layer_id]
    l2_diff = jax.tree_util.tree_map(lambda x, y: jnp.linalg.norm(x) -jnp.linalg.norm(y), wrt_layer_grad, shifted_grad)
    cos_angle = jax.tree_util.tree_map(cosine, wrt_layer_grad, shifted_grad)
    #print("l2_norm [conv 3]", l2_norm.conv.kernel.value)
    #print("cosine angle [conv 3]", cos_angle.conv.kernel.value)

    return l2_diff, cos_angle
