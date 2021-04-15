import jax.numpy as jnp
from jax import random
import jax
from typing import Union, Callable, Sequence, NamedTuple
from functools import partial


class COO(NamedTuple):
    """Defines a namedtuple for coordinate list sparse matrix (COO)."""

    rows: jnp.ndarray
    cols: jnp.ndarray
    vals: jnp.ndarray
    shape: Sequence[int]


def coo_matrix(
    key: jnp.ndarray,
    shape: Sequence[int],
    sparsity: Union[float],
    init_fn: Callable = random.normal,
):
    """ Constructs sparse matrix of given shape and sparsity level with init_fn values 
    in the coordinate list form. Only works for 2D matrices."""

    # Calculating shapes and stuff
    vals_key, rows_key, cols_key = random.split(key, 3)
    n_elements = shape[0] * shape[1]
    n_non_zero = jnp.around(n_elements * sparsity).astype(jnp.int32)

    # Creating matrix
    vals = init_fn(key=vals_key, shape=(n_non_zero,))
    rows = random.choice(rows_key, shape[0], (n_non_zero,))
    cols = random.choice(cols_key, shape[1], (n_non_zero,))
    return COO(rows, cols, vals, shape)


def coo_to_dense(sparse_mat: COO):
    """Changes sparse matrix of COO from to dense matrix."""
    dense_mat = jnp.zeros(sparse_mat.shape, dtype=sparse_mat.vals.dtype)
    dense_mat = jax.ops.index_update(
        dense_mat, (sparse_mat.rows, sparse_mat.cols), sparse_mat.vals
    )
    return dense_mat


def coomat_vec_mul(sparse_mat: COO, vec: jnp.ndarray):
    """Performs COO sparse matrix - dense vector multiplication"""
    return _coomat_vec_mul(sparse_mat, vec, sparse_mat.shape[0])


@partial(jax.jit, static_argnums=(4,))
def _coomat_vec_mul(rows, cols, vals, vec, shape):
    """Performs COO matrix - dense vector multiplication"""
    prod = jnp.take(vec, cols, axis=0) * vals[:, None]
    return jax.ops.segment_sum(prod, rows, shape)
