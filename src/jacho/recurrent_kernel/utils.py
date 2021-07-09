from jax import numpy as jnp
from typing import Callable


def weighted_dot(sigma_i: float, sigma_r: float, sigma_b: float) -> Callable:
    """Defines weighed dot product, i.e. <u, v> with u = [i, j]
    Returns function which calculates the product given the weights.
    """

    def dot(gram: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        return sigma_i ** 2 * gram + sigma_r ** 2 * kernel + sigma_b ** 2

    return dot
