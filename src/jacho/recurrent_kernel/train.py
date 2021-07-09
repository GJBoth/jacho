import jax
import jax.numpy as jnp


def train(model, params, data, *, alpha):
    K_train, model_state = model.apply(params, data, method=model.train_kernel)
    K_train = jax.ops.index_add(K_train, jnp.diag_indices(K_train.shape[0]), alpha)

    # Calculating W_out; maybe use SVD through linalg?
    c, low = jax.scipy.linalg.cho_factor(K_train, check_finite=False)
    W_out = jax.scipy.linalg.cho_solve(
        (c, low), data[-K_train.shape[0] :], check_finite=False
    )
    return (W_out, *model_state)
