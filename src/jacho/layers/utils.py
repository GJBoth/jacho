from flax import linen as nn
from jax import random, numpy as jnp
import numpy as np
from typing import Callable
from functools import reduce


class Diagonal(nn.Module):
    @nn.compact
    def __call__(self, X):
        D = self.param("kernel", random.rademacher, (1, X.shape[1]))
        return D * X


def hadamard(normalized=True, dtype=jnp.float32):
    """ We need the numpy to use it as initializer"""

    def init(key, shape, dtype=dtype):
        n = shape[0]
        # Input validation
        if n < 1:
            lg2 = 0
        else:
            lg2 = np.log2(n)
        assert 2 ** lg2 == n, "shape must be a positive integer and a power of 2."

        # Logic
        H = jnp.ones((1,), dtype=dtype)
        for i in np.arange(lg2):
            H = jnp.vstack([jnp.hstack([H, H]), jnp.hstack([H, -H])])

        if normalized:
            H = 2 ** (-lg2 / 2) * H
        return H

    return init


class HadamardTransform(nn.Module):
    n_hadamard: int

    def setup(self):
        self.H = hadamard()(None, (self.n_hadamard,))

    def __call__(self, X):
        return jnp.dot(X, self.H)


class Log2Padding(nn.Module):
    padding_fn: Callable = jnp.zeros

    @nn.compact
    def __call__(self, X):
        n_in = X.shape[-1]  # finding next power of 2
        next_power = (2 ** np.ceil(np.log2(n_in))).astype(np.int32)
        n_padding = (next_power - n_in).astype(np.int32)
        return jnp.concatenate([X, self.padding_fn((1, n_padding))], axis=-1)


class FastHadamardTransform(nn.Module):
    @nn.compact
    def __call__(self, X):
        def update(z, m):
            x = z[:, ::2, :]
            y = z[:, 1::2, :]
            return jnp.concatenate((x + y, x - y), axis=-1)

        m_max = np.log2(X.shape[-1])
        X = jnp.expand_dims(X, -1)
        X = reduce(update, np.arange(m_max), X)
        return X.squeeze(-2) / 2 ** (m_max / 2)
