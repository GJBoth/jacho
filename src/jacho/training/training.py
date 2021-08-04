from flax.core import unfreeze
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from flax.core import freeze, unfreeze
import jax


def ridge(model, params, state, u_train, renorm_factor=1.0, alpha=1e-2, n_init=50):
    # Running reservoir and constructing feature matrix
    updated_state, intermediate = model.apply(
        params, state, u_train[:-1], method=model.run_reservoir
    )
    X = jnp.concatenate(
        [intermediate[n_init - 1 :], renorm_factor * u_train[n_init - 1 : -1]], axis=-1
    ).squeeze()

    # Solving for output kernel
    # TODO: Check why this is pretty slow?
    c, low = cho_factor(
        jax.ops.index_add(jnp.dot(X.T, X), jnp.diag_indices(X.shape[1]), alpha)
    )
    W_out = cho_solve((c, low), jnp.dot(X.T, u_train[n_init:].squeeze()))

    # Updating outputlayer in parameters
    # TODO: Change to jax traverse_util
    params = unfreeze(params)
    params["params"]["output_layer"]["Dense_0"]["kernel"] = W_out
    params = freeze(params)

    return updated_state, params
