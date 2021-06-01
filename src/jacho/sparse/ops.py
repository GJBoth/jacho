from typing import Union
from .COO import COO, coo_to_dense


def sparse_to_dense(
    sparse_mat: Union[COO,]
):
    """Transforms sparse matrix of any type to dense. Selects right conversion
    depending on type of sparse matrix and then converts."""
    if type(sparse_mat).__name__ == "COO":
        to_dense_fn = coo_to_dense
    else:
        raise NotImplementedError
    return to_dense_fn(sparse_mat)

