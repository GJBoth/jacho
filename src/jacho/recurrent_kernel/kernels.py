from jax import numpy as jnp


def erf_kernel(uv: jnp.ndarray, uu: jnp.ndarray, vv: jnp.ndarray) -> jnp.ndarray:
    """Defines kernel corresponding to erf update function."""
    z = 2 * uv / jnp.sqrt((1 + 2 * uu) * (1 + 2 * vv))
    return 2 / jnp.pi * jnp.arcsin(z)
