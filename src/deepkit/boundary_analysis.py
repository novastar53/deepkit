"""
Utilities for analyzing decision boundaries in neural networks.

Focuses on measuring boundary alignment, margins, and geometry.
"""

from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax


def compute_decision_margin(
    network: nnx.Module,
    X: jnp.ndarray,
    y: jnp.ndarray,
    num_steps: int = 50
) -> float:
    """
    Compute minimum distance from decision boundary to training points.

    The decision margin is the minimum distance from any training point
    to the decision boundary. Larger margins generally indicate better
    generalization.

    Args:
        network: Neural network classifier
        X: Training data of shape (N, ...)
        y: Training labels
        num_steps: Steps for binary search to find boundary

    Returns:
        Minimum distance to decision boundary across all samples
    """
    margins = []

    for i in range(len(X)):
        xi, yi = X[i:i+1], y[i:i+1]

        # Find closest point on decision boundary using gradient-based search
        margin = _distance_to_boundary(network, xi, yi, num_steps)
        margins.append(margin)

    return float(jnp.min(jnp.array(margins)))


def _distance_to_boundary(
    network: nnx.Module,
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_steps: int
) -> float:
    """
    Find distance from point x to decision boundary using binary search.

    Searches along gradient direction from point toward decision boundary.
    """
    def loss_fn(model, x, y):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    # Compute gradient (points toward increasing loss, i.e., away from class)
    grad_fn = jax.grad(lambda x: loss_fn(network, x, y))
    grad = grad_fn(x)

    # Normalize gradient direction
    grad_norm = jnp.linalg.norm(grad)
    if grad_norm < 1e-10:
        return 0.0
    direction = grad / grad_norm

    # Binary search for boundary (where prediction changes)
    low = 0.0
    high = 2.0  # Max search distance

    for _ in range(num_steps):
        mid = (low + high) / 2.0
        test_point = x + mid * direction

        logits = network(test_point)
        pred = jnp.argmax(logits, axis=-1)

        if pred == y:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def compute_boundary_alignment(
    network: nnx.Module,
    X: jnp.ndarray,
    y: jnp.ndarray
) -> Tuple[float, float]:
    """
    Measure how well decision boundaries align with training data.

    Computes two metrics:
    1. Mean distance from data points to decision boundary
    2. Boundary density around data (boundaries per unit volume near data)

    Args:
        network: Neural network classifier
        X: Training data
        y: Training labels

    Returns:
        Tuple of (mean_distance, boundary_density)
    """
    distances = []

    for i in range(len(X)):
        xi, yi = X[i:i+1], y[i:i+1]

        dist = _distance_to_boundary(network, xi, yi, num_steps=30)
        distances.append(dist)

    mean_distance = float(jnp.mean(jnp.array(distances)))

    # Boundary density: approximate by checking nearby prediction stability
    # Sample subset for efficiency
    num_samples = min(50, len(X))
    sample_indices = jax.random.choice(
        jax.random.PRNGKey(0),
        len(X),
        shape=(num_samples,),
        replace=False
    )

    boundary_count = 0
    total_checked = 0

    for idx in sample_indices:
        xi = X[idx:idx+1]
        yi = y[idx:idx+1]

        # Check prediction stability in small neighborhood
        key = jax.random.PRNGKey(idx)
        noise = jax.random.uniform(key, shape=(5, xi.shape[-1])) - 0.5
        noise = noise * 0.1  # Small perturbations

        perturbed_points = xi + noise
        logits = network(perturbed_points)
        preds = jnp.argmax(logits, axis=-1)

        # Count how many predictions differ from original
        original_pred = jnp.argmax(network(xi), axis=-1)
        boundary_count += jnp.sum(preds != original_pred)
        total_checked += 5

    boundary_density = boundary_count / total_checked if total_checked > 0 else 0.0

    return mean_distance, float(boundary_density)


def compute_gradient_norm_statistics(
    network: nnx.Module,
    X: jnp.ndarray,
    y: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute gradient norm statistics to analyze loss landscape smoothness.

    Args:
        network: Neural network
        X: Training data
        y: Training labels

    Returns:
        Dictionary with gradient norm statistics
    """
    def loss_fn(model, x, y):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    # Compute average gradient norm across dataset
    total_grad_norm = 0.0
    num_samples = min(20, len(X))

    for i in range(num_samples):
        xi, yi = X[i:i+1], y[i:i+1]

        grad_fn = jax.grad(lambda x: loss_fn(network, x, yi))
        grads = grad_fn(xi)

        # Compute L2 norm
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        total_grad_norm += float(grad_norm)

    return {
        'mean_grad_norm': total_grad_norm / num_samples,
    }
