import jax
import jax.numpy as jnp
import flax.nnx as nnx


def get_attention_map(q, k):

    a = q @ k / jnp.sqrt(q.shape[-1])