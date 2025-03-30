import copy

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx


def mask_layers(layer_prefix, params):
    state = {"prev": True}
    def mask_fn(path, leaf):
        if state["prev"] is False: 
            return 0
        path = ".".join([str(p.key) for p in path if isinstance(p, jax.tree_util.DictKey)])
        if layer_prefix in path:
            state["prev"] = False 
            return 0
        return leaf

    return jax.tree_util.tree_map_with_path(mask_fn, params)


#@nnx.jit(static_argnums=[4,])
def partial_step_fn(model: nnx.Module, optimizer: nnx.Optimizer, batch: jax.Array, labels: jax.Array, wrt_layer: str):

    def loss_fn(model, batch, targets):
        logits, activations = model(batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss, activations
    
    # create copies of the model and optimizer 
    optimizer_copy = optimizer.__deepcopy__()

    (loss, activations), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch, labels)

    # update the weights of the previous layers
    print("model param")
    print(model.convs[0].conv.kernel.value[0, 0, 0, 0])
    print("optimizer model param")
    print(optimizer_copy.model.convs[0].conv.kernel.value[0, 0, 0, 0])
    print("optimizer state")
    print(optimizer_copy.opt_state[0].trace.convs[0].conv.kernel[0, 0, 0, 0])
    print("original optimizer state")
    print(optimizer.opt_state[0].trace.convs[0].conv.kernel[0, 0, 0, 0])
    # update the previous layers
    partial_grads = mask_layers(wrt_layer, grads)
    print("gradient")
    print(partial_grads.convs[0].conv.kernel.value[0, 0, 0, 0])
    optimizer_copy.update(partial_grads)
    print("updated model param")
    print(model.convs[0].conv.kernel.value[0, 0, 0, 0])
    print("updated optimizer model param")
    print(optimizer_copy.model.convs[0].conv.kernel.value[0, 0, 0, 0])
    print("original optimizer model param")
    print(optimizer.model.convs[0].conv.kernel.value[0, 0, 0, 0])
    print("updated optimizer velocity")
    print(optimizer_copy.opt_state[0].trace.convs[0].conv.kernel.value[0, 0, 0, 0])
    print("original optimizer velocity")
    print(optimizer.opt_state[0].trace.convs[0].conv.kernel.value[0, 0, 0, 0])
    # calculate the shifted loss and gradients
    (shifted_loss, _), shifted_grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer_copy.model, batch, labels)
    # pick out the gradient wrt current layer
    wrt_layer_grad = grads["convs"][3] 
    shifted_grad = shifted_grads["convs"][3]
    l2_norm = jax.tree_util.tree_map(lambda x, y: jnp.linalg.norm(x - y), wrt_layer_grad, shifted_grad)
    print("l2_norm", l2_norm)

    return loss, activations, shifted_grads
