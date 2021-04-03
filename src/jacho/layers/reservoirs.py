from types import new_class
from typing import Callable, Tuple
from flax import linen as nn
from flax.linen.initializers import normal, zeros
from .activation import leaky_erf
import jax.numpy as jnp
from .utils import Diagonal, HadamardTransform, FastHadamardTransform, Log2Padding
import numpy as np


class RandomReservoir(nn.Module):
    """ Implements a generic reservoir."""

    n_reservoir: int
    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    activation_fn: Callable = leaky_erf
    activation_fn_args: Tuple = ()

    @nn.compact
    def __call__(self, state, x):
        # TODO: Turn state into flax variable.
        z_input = nn.Dense(
            self.n_reservoir,
            kernel_init=normal(self.input_scale),
            bias_init=normal(self.bias_scale),
        )
        z_res = nn.Dense(
            self.n_reservoir, kernel_init=normal(self.res_scale), use_bias=False
        )
        updated_state = self.activation_fn(
            z_input(x) + z_res(state), state, *self.activation_fn_args
        )

        return updated_state

    @staticmethod
    def initialize_state(rng, n_reservoir, init_fn=zeros):
        return init_fn(rng, (1, n_reservoir))


class StructuredTransform(nn.Module):
    n_reservoir: int

    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    activation_fn: Callable = leaky_erf
    activation_fn_args: Tuple = (1.0,)

    n_layers: int = 3

    @nn.compact
    def __call__(self, state, inputs):
        X = jnp.concatenate([self.res_scale * state, self.input_scale * inputs], axis=1)
        X = Log2Padding()(X)  # automatically pad to next power of 2
        hadamard = HadamardTransform(X.shape[-1])
        for _ in jnp.arange(self.n_layers):
            X = hadamard(Diagonal()(X))

        bias = self.param("bias", normal(stddev=self.bias_scale), (self.n_reservoir,))
        # TODO: check if self.n_hadamard is correct; comes from code from paper
        X = X[:, : self.n_reservoir] / X.shape[-1] + bias
        X = self.activation_fn(X, state, *self.activation_fn_args)
        return X

    @staticmethod
    def initialize_state(rng, n_reservoir, init_fn=zeros):
        return init_fn(rng, (1, n_reservoir))


class FastStructuredTransform(nn.Module):
    n_reservoir: int

    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    activation_fn: Callable = leaky_erf
    activation_fn_args: Tuple = (1.0,)

    n_layers: int = 3

    @nn.compact
    def __call__(self, state, inputs):
        X = jnp.concatenate([self.res_scale * state, self.input_scale * inputs], axis=1)
        X = Log2Padding()(X)  # automatically pad to next power of 2
        hadamard = FastHadamardTransform()
        for _ in jnp.arange(self.n_layers):
            X = hadamard(Diagonal()(X))

        bias = self.param("bias", normal(stddev=self.bias_scale), (self.n_reservoir,))
        # TODO: check if self.n_hadamard is correct; comes from code from paper
        X = X[:, : self.n_reservoir] / X.shape[-1] + bias
        X = self.activation_fn(X, state, *self.activation_fn_args)
        return X

    @staticmethod
    def initialize_state(rng, n_reservoir, init_fn=zeros):
        return init_fn(rng, (1, n_reservoir))
