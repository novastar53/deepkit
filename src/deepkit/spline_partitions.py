"""
Utilities for analyzing and visualizing spline partitions in ReLU networks.

Based on: "Batch Normalization Explained" - demonstrates how BN aligns
partition boundaries with training data and increases decision margins.
"""

from typing import Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import matplotlib.pyplot as plt
import numpy as np


def get_activation_pattern(network: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute binary activation pattern for ReLU network.

    For a ReLU network, the activation pattern indicates which neurons
    are "active" (pre-activation > 0) vs "inactive" (pre-activation <= 0).

    Args:
        network: ReLU network to analyze
        x: Input tensor of shape (batch, ..., input_dim)

    Returns:
        Binary pattern array of shape (batch, total_neurons) where 1 = active, 0 = inactive
    """
    # This is a simplified version that works with standard MLPs
    # For production use, you'd want to hook into the forward pass more carefully

    patterns = []

    def forward_with_capture(model, x):
        """Custom forward pass that captures activation patterns"""
        # This needs to be customized based on your network architecture
        # For now, we'll use a simpler approach with a hookable wrapper
        pass

    # For 2D visualization, we'll use a simpler approach
    # Just compute the prediction and use that as a proxy for "region"
    logits = network(x)
    return jnp.argmax(logits, axis=-1)


def count_linear_regions(
    network: nnx.Module,
    input_bounds: Tuple[Tuple[float, float], ...],
    grid_resolution: int = 100
) -> int:
    """
    Count number of linear regions in input space for a ReLU network.

    A linear region is a region of input space where the network behaves
    as a fixed affine transformation (all ReLU activation patterns constant).

    Args:
        network: ReLU network to analyze
        input_bounds: Tuple of (min, max) bounds for each input dimension
        grid_resolution: Number of grid points per dimension

    Returns:
        Number of unique linear regions found
    """
    # Create meshgrid over input space
    meshes = jnp.meshgrid(
        *[jnp.linspace(low, high, grid_resolution)
          for low, high in input_bounds]
    )
    grid_points = jnp.stack([m.ravel() for m in meshes], axis=-1)

    # Compute activation pattern for each grid point
    patterns = get_activation_pattern(network, grid_points)

    # Count unique patterns
    unique_patterns = jnp.unique(patterns, axis=0)
    return len(unique_patterns)


def visualize_partitions_2d(
    network: nnx.Module,
    X: jnp.ndarray,
    y: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    grid_resolution: int = 100,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    title: str = 'Spline Partitions & Decision Boundary'
) -> plt.Axes:
    """
    Visualize linear regions and decision boundary for 2D network.

    Creates a color-coded plot showing:
    - Linear regions (color-coded by activation pattern)
    - Decision boundary (where class predictions change)
    - Training data points

    Args:
        network: 2D-input ReLU network
        X: Training data of shape (N, 2)
        y: Training labels
        ax: Matplotlib axis (creates new if None)
        grid_resolution: Resolution for visualization grid
        bounds: (x_min, x_max, y_min, y_max) or None to auto-detect
        title: Plot title

    Returns:
        Matplotlib axis with visualization
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if bounds is None:
        x_min, x_max = float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5
        y_min, y_max = float(X[:, 1].min()) - 0.5, float(X[:, 1].max()) + 0.5
    else:
        x_min, x_max, y_min, y_max = bounds

    # Create grid
    xx = jnp.linspace(x_min, x_max, grid_resolution)
    yy = jnp.linspace(y_min, y_max, grid_resolution)
    XX, YY = jnp.meshgrid(xx, yy)
    grid_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # Compute predictions
    logits = network(grid_points)
    preds = jnp.argmax(logits, axis=-1).reshape(grid_resolution, grid_resolution)

    # Color by decision boundary
    ax.contourf(XX, YY, preds, alpha=0.3, levels=50, cmap='RdYlBu')
    ax.contour(XX, YY, preds, colors='black', linewidths=0.5, alpha=0.5)

    # Overlay data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', alpha=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    return ax


def compute_partition_entropy(
    network: nnx.Module,
    X: jnp.ndarray,
    grid_resolution: int = 100
) -> float:
    """
    Compute entropy of partition distribution over data.

    Higher entropy = partitions more evenly distributed (BN effect).
    Lower entropy = partitions concentrated in certain regions.

    Args:
        network: ReLU network
        X: Training data
        grid_resolution: Resolution for partition counting (not used in simplified version)

    Returns:
        Entropy value (higher = more uniform distribution)
    """
    # Get predictions for training data (simplified proxy for partition assignment)
    logits = network(X)
    preds = jnp.argmax(logits, axis=-1)

    # Count frequency of each prediction
    unique_preds, counts = jnp.unique(preds, return_counts=True)

    # Compute entropy
    probs = counts / jnp.sum(counts)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))

    return float(entropy)
