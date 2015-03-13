import numpy as np


U_FILE_PATH = 'results/after_factorization/u.csv'
V_FILE_PATH = 'results/after_factorization/v.csv'


def read_data():
    """ Read data from u, v
    """
    u = np.genfromtxt(U_FILE_PATH, delimiter=',')
    v = np.genfromtxt(V_FILE_PATH, delimiter=',')
    return u, v


def project(u, v, dimensions=2):
    """Factor all matrices with SVD and project them to the specified number of dimensions

    :param np.array u: first matrix to project
    :param np.array v: second matrix to project
    :param int dimensions: new dimensionality of the matrices
    """
    a, _, _ = np.linalg.svd(v)
    u_new = np.dot(np.transpose(a[:, :dimensions]), np.transpose(u))
    v_new = np.dot(np.transpose(a[:, :dimensions]), v)
    return u_new, v_new


def run():
    u, v = read_data()
    u2, v2 = project(u, v)
    np.savetxt("results/after_svd/u_2dim.csv", u2, delimiter=',')
    np.savetxt("results/after_svd/v_2dim.csv", v2, delimiter=',')


if __name__ == '__main__':
    run()
