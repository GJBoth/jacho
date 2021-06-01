from flax import linen as nn
from functools import partial
from flax.linen.initializers import zeros
import jax.numpy as jnp
from typing import Tuple

# Shortcut for scan to run flax model
scan = partial(
    nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
)


class GenericEchoState(nn.Module):
    n_reservoir: int
    reservoir_type: nn.Module
    reservoir_args: Tuple

    n_out: int
    output_layer_type: nn.Module
    output_layer_args: Tuple

    def setup(self):
        self.reservoir = self.reservoir_type(self.n_reservoir, *self.reservoir_args)
        self.output_layer = self.output_layer_type(self.n_out, *self.output_layer_args)

    def __call__(self, state, initial, n_steps=1):
        return self.predict((state, initial), jnp.arange(n_steps))

    @scan
    def predict(self, carry, _):
        state, previous_prediction = carry

        # Getting new reservoir state
        updated_state = self.reservoir(state, previous_prediction)

        # Calculating output layer
        prediction = self.output_layer(updated_state, previous_prediction)
        return (updated_state, prediction), prediction

    @scan
    def run_reservoir(self, state, inputs):
        # Getting new reservoir state
        updated_state = self.reservoir(state, inputs)

        # Scan; first output gets carried, second gets saved
        return updated_state, updated_state

    def initialize_state(self, rng, n_reservoir, init_fn=zeros):
        return self.reservoir_type.initialize_state(rng, n_reservoir, init_fn)

