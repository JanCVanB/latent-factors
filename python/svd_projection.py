import numpy as np

U_FILE_PATH = 'results/after_factorization/u.csv'
V_FILE_PATH = 'results/after_factorization/v.csv'


def read_data():
    """ Read data from u, v
    """
    u = np.genfromtxt(U_FILE_PATH, delimiter=',')
    v = np.genfromtxt(V_FILE_PATH, delimiter=',')
   
    return u, v


def project(dimensions, *matrices):
    """Factor all matrices with SVD and project them to the specified number of dimensions

    :param int dimensions: new dimensionality of the matrices
    :param tuple matrices: matrices to project
    """
    projected_matrices = []
    for matrix in matrices:
        a, _, _ = np.linalg.svd(matrix)
        # TODO: determine if old code was correct (both av? transpose u?)
        # u2 = np.dot(np.transpose(av[:, :2]), np.transpose(u))
        # v2 = np.dot(np.transpose(av[:, :2]), v)
        projected_matrix = np.dot(np.transpose(a[:, :dimensions]), matrix)
        projected_matrices.append(projected_matrix)
    return tuple(projected_matrices)


def run():
    u, v = read_data()
    u2, v2 = project(2, u, v)
    np.savetxt("results/after_svd/u_2dim.csv", u2, delimiter=',')
    np.savetxt("results/after_svd/v_2dim.csv", v2, delimiter=',')


if __name__ == '__main__':
    run()
