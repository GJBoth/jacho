from typing import Callable, Tuple
import jax.numpy as jnp
from jax.lax import scan
from functools import partial
from jax import jit
import jax

# ============================ Kernels ==========================================
def erf_kernel(uv: jnp.ndarray, uu: jnp.ndarray, vv: jnp.ndarray) -> jnp.ndarray:
    """Defines kernel corresponding to erf update function."""
    z = 2 * uv / jnp.sqrt((1 + 2 * uu) * (1 + 2 * vv))
    return 2 / jnp.pi * jnp.arcsin(z)


# ============================== Utils ==========================================
def weighted_dot(sigma_i: float, sigma_r: float, sigma_b: float) -> Callable:
    """Defines weighed dot product, i.e. <u, v> with u = [i, j]
    Returns function which calculates the product given the weights.
    """

    def dot(gram: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        return sigma_i ** 2 * gram + sigma_r ** 2 * kernel + sigma_b ** 2

    return dot


# ============================== Update functions ==========================================
def update_fn(kernel_fn: Callable, dot_fn: Callable, uu: jnp.ndarray) -> Callable:
    """Update function to scan across rows to generate kernel matrix."""

    def update(
        k_prev: jnp.ndarray, inputs: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gram, vv = inputs
        # we shift and pad with zero as were moving in time.
        k_prev = jnp.pad(k_prev[:-1], (1, 0))
        uv = dot_fn(gram, k_prev)
        k = kernel_fn(uv, uu, vv)
        return k, k

    return update


def diagonal_update_fn(kernel_fn: Callable, dot_fn: Callable) -> Callable:
    """Update function to scan across diagonal of gram matrix to generate
    diagonal of kernel matrix.
    """

    def update_fn(k_prev, gram) -> Tuple[jnp.ndarray, jnp.ndarray]:
        uv = dot_fn(gram, k_prev)
        k = kernel_fn(uv, uv, uv)  # on diagonal uv=uu=vv
        return k, k

    return update_fn


def predict_fn(data_train, kernel_fn, dot_fn, uu, W_out, n_init, renorm):
    def predict(carry, _):
        k_prev, k_diag_prev, data_test = carry

        # Calculating grams
        gram_diag_prev = jnp.dot(data_test, data_test.T)
        gram_prev = jnp.dot(data_test, data_train.T)

        # Updating test/test diagonal
        vv = dot_fn(gram_diag_prev, k_diag_prev)
        k_diag = kernel_fn(vv, vv, vv)

        # Updating test/train kernel
        k_prev = jnp.pad(k_prev[:-1], (1, 0))
        uv = dot_fn(gram_prev, k_prev)
        k = kernel_fn(uv, uu, vv)

        # Prediction
        prediction = jnp.dot((k + renorm * gram_prev)[n_init - 1 :], W_out)

        return (k, k_diag, prediction), (k, prediction)

    return predict


# ================== User facing functions ==================================
@partial(jit, static_argnums=(1,))
def recurrent_kernel(
    data: jnp.ndarray, kernel_fn: Callable, scaling_args: Tuple[float, float, float]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Preliminaries
    dot_fn = weighted_dot(*scaling_args)
    gram = jnp.dot(data[:-1], data[:-1].T)
    n_samples = gram.shape[0]

    # Getting diagonal results and <u, u>
    _, k_diagonal = scan(
        diagonal_update_fn(kernel_fn, dot_fn),
        0.0,
        jnp.diag(gram),
    )
    k_diagonal = jnp.pad(k_diagonal[:-1], (1, 0))
    uu = dot_fn(jnp.diag(gram), k_diagonal)

    # Running over axes
    _, K_recurrent = scan(
        update_fn(kernel_fn, dot_fn, uu),
        jnp.zeros((n_samples,)),
        (gram, uu),
    )
    # Padding with zeros for first data.
    return jnp.pad(K_recurrent, ((1, 0), (1, 0))), uu


@partial(jax.jit, static_argnums=(4,))
def train(K, data, alpha, renorm, n_init=50):
    # Need to shift cause we dont use the last of the train
    K_train = K[n_init:, n_init:]
    K_train += renorm * jnp.dot(data[n_init - 1 : -1], data[n_init - 1 : -1].T)
    K_train += alpha * jnp.eye(K_train.shape[0])

    c, low = jax.scipy.linalg.cho_factor(K_train, check_finite=False)
    W_out = jax.scipy.linalg.cho_solve((c, low), data[n_init:], check_finite=False)

    return W_out


@partial(jax.jit, static_argnums=(1, 2, 6, 8))
def predict(data, kernel, dot_fn, uu, K_train, W_out, n_init, renorm, length):
    rec_pred_fn = predict_fn(
        data[:-1], kernel, dot_fn, uu, W_out, n_init=n_init, renorm=renorm
    )
    _, (K_test, prediction) = jax.lax.scan(
        rec_pred_fn,
        (K_train[-1, 1:], K_train[-1, -1], data[-1]),
        None,
        length=length,
    )

    return prediction, jnp.pad(K_test.squeeze(), ((0, 0), (1, 0)))
