from typing import Callable, Tuple
from flax import linen as nn
from jax import numpy as jnp
from jax.lax import scan
import jax
from .utils import weighted_dot

Tensor = jnp.ndarray


class RecurrentKernel(nn.Module):
    kernel_fn: Callable
    n_init: int
    renorm: float
    scaling_args: Tuple[float, float, float]

    def setup(self):
        self.dot_fn = weighted_dot(*self.scaling_args)

    def __call__(self, data: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        return self.train_kernel(data)

    def train_kernel(
        self, data: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        gram = jnp.dot(data[:-1], data[:-1].T)

        # Getting diagonal
        k_diagonal_final, k_diagonal = self._train_kernel_diagonal(jnp.diag(gram))
        uu = self.dot_fn(jnp.diag(gram), k_diagonal)

        # Iterating over rows
        k_final, k_train = self._train_kernel(gram, uu)

        # Building training matrix; cutting off init
        # already we miss the init (0) so n_init -1
        k_train = k_train[self.n_init - 1 :, self.n_init - 1 :]
        k_train += self.renorm * gram[self.n_init - 1 :, self.n_init - 1 :]

        return k_train, (uu, k_final, k_diagonal_final)

    def predict(self, data_train: Tensor, model_state, *, length: int):
        def update(carry, _):
            W_out, uu, k_prev, k_diag_prev, data_test = carry

            # Calculating grams
            gram_diag_prev = jnp.dot(data_test, data_test.T)
            gram_prev = jnp.dot(data_test, data_train[:-1].T)

            # Updating test/test diagonal
            vv = self.dot_fn(gram_diag_prev, k_diag_prev)
            k_diag = self.kernel_fn(vv, vv, vv)

            # Updating test/train kernel
            k_prev = jnp.pad(k_prev[:-1], (1, 0))
            uv = self.dot_fn(gram_prev, k_prev)
            k = self.kernel_fn(uv, uu, vv)

            # Prediction
            k_predict = (k + self.renorm * gram_prev)[self.n_init - 1 :]
            prediction = jnp.dot(k_predict, W_out)
            return (W_out, uu, k, k_diag, prediction), prediction

        prediction = scan(
            update,
            (*model_state, data_train[-1]),
            None,
            length=length,
        )[1]

        return prediction

    def _train_kernel_diagonal(self, diag_gram: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculates diagonal of kernel."""

        def update(k_prev: Tensor, gram: Tensor) -> Tuple[Tensor, Tensor]:
            uv = self.dot_fn(gram, k_prev)
            k = self.kernel_fn(uv, uv, uv)  # on diagonal uv=uu=vv
            return k, k

        k_final, k_diagonal = scan(update, 0.0, diag_gram)
        k_diagonal = jnp.pad(k_diagonal[:-1], (1, 0))
        return k_final, k_diagonal

    def _train_kernel(
        self,
        gram: Tensor,
        uu: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        def update(
            k_prev: Tensor, inputs: Tuple[Tensor, Tensor]
        ) -> Tuple[Tensor, Tensor]:
            gram, vv = inputs

            # we shift and pad with zero as were moving in time.
            k_prev = jnp.pad(k_prev[:-1], (1, 0))
            uv = self.dot_fn(gram, k_prev)
            k = self.kernel_fn(uv, uu, vv)
            return k, k

        k_final, k = scan(update, jnp.zeros((gram.shape[0],)), (gram, uu))
        return k_final, k
