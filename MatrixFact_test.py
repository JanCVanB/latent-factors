import csv
import numpy as np
import math
# TODO: update e cutoff, iterations, lambda, and ita values (they caused floating-point calculation RuntimeWarning)
# TODO: rename "user_movie_matrix"

RATINGS_FILE_PATH = 'data.txt'
MOVIES_FILE_PATH = 'movies.txt'
NUM_USERS = 943


# TODO: rename "eij", and "error"
def matrix_factorization(user_movie_matrix, dimensions, iterations=1000, lambda_=0.02, learning_rate=0.0002):
    """ Matrix Factorization with missing values using gradient descent

    :param user_movie_matrix: input matrix to factorize and from which to learn the latent factor model
    :param dimensions: the "free" dimension of the latent factor model
    :param iterations: the maximum number of iterations to perform gradient descent
    :param lambda_: the regularization parameter
    :param learning_rate: the gradient descent learning rate
    :return: two latent factor models with the shapes M*dimensions and dimensions*N;
    :rtype: tuple 
    """
    m, n = user_movie_matrix.shape
    u = np.random.rand(m, dimensions)
    v = np.random.rand(dimensions, n)
    for iteration in xrange(iterations):
        print iteration
        W = np.array([[int(x > 0) for x in row] for row in user_movie_matrix])
        # learning_rate = learning_rate/math.sqrt(iteration+1)
        for i_user in xrange(m):
            error = user_movie_matrix[i_user,:] - np.dot(u[i_user, :], v)
            u[i_user,:] -= ((lambda_ * u[i_user,:] - 2 * np.dot( (W[i_user,:] * error), np.transpose(v)) ) * learning_rate)

        for j_movie in xrange(n):
            error = user_movie_matrix[:,j_movie] - np.dot(u, v[:,j_movie])
            v[:,j_movie] -= ((lambda_ * v[:,j_movie] - 2 * np.dot( np.transpose(u), (W[:,j_movie] * error)) ) * learning_rate) 
    return  u, v



def read_data(ratings_file_path, movies_file_path):
    """ Read data from files containing user-movie ratings and movie tags and return a user-movie matrix

    The ratings file format is user id, movie id, rating (on each line)
    The user-movie matrix is based on rating data, where 0 indicates missing rating data

    :param str ratings_file_path: the path to the ratings data file
    :param str movies_file_path: the path to movies data file
    :return: user_movie_matrix: each row represents a user, each column represents a movie
    :rtype: numpy.array
    """
    with open(ratings_file_path, 'rU') as ratings_file:
        ratings_reader = csv.reader(ratings_file, dialect=csv.excel_tab)
        ratings = np.array([[int(x) for x in rating] for rating in ratings_reader])
    with open(movies_file_path, 'rU') as movie_file:
        movie_tags = np.array(list(csv.reader(movie_file, dialect=csv.excel_tab)))
    num_movies = movie_tags.shape[0]
    user_movie_matrix = np.zeros((NUM_USERS, num_movies), dtype=np.int8)
    for user_id, movie_id, rating in ratings:
        user_movie_matrix[user_id - 1, movie_id - 1] = rating
    return user_movie_matrix


def run():
    user_movie_matrix = read_data(RATINGS_FILE_PATH, MOVIES_FILE_PATH)
    # np.savetxt("user_movie_matrix_1.csv",user_movie_matrix,delimiter=',')
    test_matrix = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    u, v = matrix_factorization(user_movie_matrix, dimensions=20)
    np.savetxt("u.csv",u,delimiter=',')
    np.savetxt("v.csv",v,delimiter=',')

    ret = np.dot(u, v)    
    print ret
    np.savetxt("recover.csv",ret,delimiter=',')


if __name__ == '__main__':
    run()
