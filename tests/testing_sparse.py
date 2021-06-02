from jacho.layers.reservoirs import SparseReservoir
from jax import random, numpy as jnp
from jacho.layers.reservoirs.sparse import sparse_coo_init
from jax.experimental import sparse_ops

key = random.PRNGKey(42)

# Making fake data
size = 1000
x = random.normal(key, (1, 100))

"""
# Initializng model
model = SparseReservoir(size)
state = model.initialize_state(key, size)
params = model.init(key, state, x)
# Running model
updated_state = model.apply(params, state, x)
"""

M_coo = sparse_coo_init(random.normal, 0.05)(key, (10, 10))
M_dense = sparse_ops.coo_todense(*M_coo, shape=(10, 10))
M_csr = sparse_ops.csr_fromdense(M_dense, nnz=5)


def _coo_to_csr(row, nrows):
    indptr = jnp.zeros(nrows + 1, row.dtype)
    return indptr.at[1:].set(jnp.cumsum(jnp.bincount(row, length=nrows)))


print(M_csr)
