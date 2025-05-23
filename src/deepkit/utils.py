import random
import string
from datetime import datetime

import jax 
import jax.numpy as jnp
import flax.nnx as nnx


def layer_l2_norm(layer: nnx.Module):
    """Calculates the L2 norms for a tree of layer params"""
    state = nnx.state(layer)
    l2_norms = jax.tree_util.tree_map(lambda x: jnp.sqrt(jnp.sum(jnp.square(x.flatten()))), state)
    return l2_norms


def layer_l2_diff(layer1: nnx.Module, layer2: nnx.Module):
    """Calculates the L2 differences between two trees of layer params with identical structures"""
    state1 = nnx.state(layer1)
    state2 = nnx.state(layer2)
    l2_diff = jax.tree_util.tree_map(lambda x, y: jnp.sqrt(jnp.sum(jnp.square(x.flatten() - y.flatten()))), state1, state2)
    return l2_diff


def global_l2_norm(x: nnx.statelib.State):
    """Calculates a single L2 norm for a tree of layer parameters"""
    leaves, _ = jax.tree_util.tree_flatten(x)
    squared_sum = sum([jnp.sum(jnp.square(l)) for l in leaves])
    return jnp.sqrt(squared_sum)


def cosine(x: jnp.ndarray, y: jnp.ndarray):
    """Calculates the cosine of the angle between two ndarrays"""
    x = x.flatten()
    y = y.flatten()
    k = jnp.dot(x, y)
    return k / (jnp.linalg.norm(x) * jnp.linalg.norm(y))


def timeit(f, *args, tries = 10, task = None):
    """Utility to time a function for multiple runs"""

    if task is not None:
        trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        trace_dir = f"/tmp/{trace_name}"
        print(trace_dir)

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    task is not None and jax.profiler.start_trace(trace_dir)

    for _ in range(tries):
        s = datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.now()
        outcomes_ms.append(1000*(e-s).total_seconds())
    task is not None and jax.profiler.stop_trace()

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    return average_time_ms 

