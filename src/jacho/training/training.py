from flax.core import unfreeze
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from flax.core import freeze, unfreeze


def ridge(model, params, state, u_train, renorm_factor=1.0, alpha=1e-2):
    # Running reservoir and constructing feature matrix
    updated_state, intermediate = model.apply(
        params, state, u_train, method=model.run_reservoir
    )
    X = jnp.concatenate(
        [intermediate[:-1], renorm_factor * u_train[:-1]], axis=-1
    ).squeeze()
    y = u_train[1:].squeeze()

    # Solving for output kernel
    # TODO: Check why this is pretty slow?
    c, low = cho_factor(jnp.dot(X.T, X) + alpha * jnp.eye(X.shape[1]))
    W_out = cho_solve((c, low), jnp.dot(X.T, y))

    # Updating outputlayer in parameters
    # TODO: Change to jax traverse_util
    params = unfreeze(params)
    params["params"]["output_layer"]["Dense_0"]["kernel"] = W_out
    params = freeze(params)

    return updated_state, params

