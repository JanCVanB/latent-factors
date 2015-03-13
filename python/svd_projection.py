import csv
import numpy as np
import math
from numpy.linalg import inv
from numpy import genfromtxt
# TODO: rename "user_movie_matrix"

U_FILE_PATH = 'results/after_factorization/u1.csv'
V_FILE_PATH = 'results/after_factorization/v1.csv'


def read_data(u_file_path, v_file_path):
    """ Read data from u, v
    """
    u = genfromtxt(U_FILE_PATH, delimiter=',')
    v = genfromtxt(V_FILE_PATH, delimiter=',')
   
    return u, v


def run():
    """
    compute svd of u, v 
    then projection to 2 dimention
    """
    u, v = read_data(U_FILE_PATH, V_FILE_PATH)
    # np.savetxt("user_movie_matrix_1.csv", user_movie_matrix, delimiter=',')
    
    Au, su, Bu = np.linalg.svd(u)
    Av, sv, Bv = np.linalg.svd(v)

    print Au.shape
    print Av.shape
    u2 = np.dot(np.transpose(Av[:, :2]), np.transpose(u))
    v2 = np.dot(np.transpose(Av[:, :2]), v)
    np.savetxt("results/after_svd/u_2dim.csv", u2, delimiter=',')
    np.savetxt("results/after_svd/v_2dim.csv", v2, delimiter=',')


if __name__ == '__main__':
    run()
