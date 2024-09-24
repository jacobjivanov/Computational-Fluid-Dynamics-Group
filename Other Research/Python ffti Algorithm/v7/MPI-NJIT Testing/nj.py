# https://github.com/netw0rkf10w/pyADGM
import numpy as np
import numba
import scipy.sparse

@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_inequality_sparse(data, indices, indptr, num_vectors):
    """
    simplex project (with inequality constraints) of each row or each column of a sparse matrix C
    If C is CSR: each row; if C is CSC: each column
    data, indices, indptr, shape: representation of C (same notation as scipy csr/csc matrix)
    num_vectors = number of rows if C is CSR
                = number of cols if C is CSC
    """
    x = np.zeros(len(data))
    for i in numba.prange(num_vectors):
        # projection for each row independently
        start = indptr[i]
        end = indptr[i+1]
        if end <= start:
            continue
        ci = data[start:end]
        u = np.maximum(ci, 0)
        if np.sum(u) <= 1:
            xi = u
        else:
            ni = end - start
            a = -np.sort(-ci)
            lambdas = (np.cumsum(a) - 1)/np.arange(1, ni+1)
            xi = 1
            for k in range(ni-1, -1, -1):
                if a[k] > lambdas[k]:
                    xi = np.maximum(ci - lambdas[k], 0)
                    break
        x[start:end] = xi
    return x


def test_simplex_projection():
    A = scipy.sparse.random(4, 5, density=0.5, format='csr')
    x = simplex_projection_inequality_sparse(A.data, A.indices, A.indptr, A.shape[0])
    print(x)

if __name__ == "__main__":
    test_simplex_projection()