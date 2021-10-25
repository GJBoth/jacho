from typing import Callable, Tuple
from flax import linen as nn
from flax.linen.initializers import normal, zeros
from ..activation import leaky_erf
import jax.numpy as jnp
from jax.experimental import sparse
from jax import random


class SparseReservoir(nn.Module):
    """Implements a sparse reservoir."""

    n_reservoir: int
    nnz: float = 0.1
    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    activation_fn: Callable = leaky_erf
    activation_fn_args: Tuple = ()

    @nn.compact
    def __call__(self, state, x):
        z_res = Sparse(
            self.n_reservoir,
            self.nnz,
            use_bias=True,
            kernel_init=normal(self.res_scale),
            bias_init=normal(self.bias_scale),
        )

        z_input = Sparse(
            self.n_reservoir,
            self.nnz,
            use_bias=False,
            kernel_init=normal(self.input_scale),
        )
        updated_state = self.activation_fn(
            z_input(x) + z_res(state), state, *self.activation_fn_args
        )

        return updated_state

    @staticmethod
    def initialize_state(rng, n_reservoir, init_fn=zeros):
        return init_fn(rng, (1, n_reservoir))


class Sparse(nn.Module):
    n_features: int
    nnz: float

    kernel_init: Callable = normal()
    use_bias: bool = True
    bias_init: Callable = normal()

    @nn.compact
    def __call__(self, inputs):
        dense_shape = (self.n_features, inputs.shape[-1])
        kernel = self.param(
            "kernel",
            sparse_coo_init(self.kernel_init, self.nnz),
            dense_shape,
        )
        # we need to transpose twice for correct shapes.
        z = sparse_ops.coo_matmat(*kernel, inputs.T, shape=dense_shape).T
        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (self.n_features,),
            )
            z += bias
        return z


def sparse_coo_init(rng, nnz):
    def _init(key, shape, dtype=jnp.float32):
        _nnz = int(nnz * shape[0] * shape[1])

        key_vals, key_rows, key_cols = random.split(key, 3)
        # Making sparse matrix in dense shape
        vals = rng(key_vals, (_nnz,), dtype)
        rows = random.choice(key_rows, shape[0], (_nnz,))
        cols = random.choice(key_cols, shape[1], (_nnz,))
        return vals, rows, cols

    return _init
