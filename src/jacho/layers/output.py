from flax import linen as nn
import jax.numpy as jnp


class Residual(nn.Module):
    n_out: int
    prediction_scale: float

    @nn.compact
    def __call__(self, state, prediction):
        z = jnp.concatenate([state, self.prediction_scale * prediction], axis=-1)
        output = nn.Dense(self.n_out, use_bias=False)(z)

        return output

