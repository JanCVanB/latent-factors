import csv
import numpy as np
from math import sqrt
# TODO: rename "user_movie_matrix"

RATINGS_FILE_PATH = 'data.txt'
MOVIES_FILE_PATH = 'movies.txt'
NUM_USERS = 943


def matrix_factorization(user_movie_matrix, dimensions=2, iterations=50, lambda_=10, learning_rate=0.01):
    """ Matrix Factorization with missing values using gradient descent

    :param numpy.array user_movie_matrix: input matrix to factorize and from which to learn the latent factor model
    :param int dimensions: the "free" dimension of the latent factor model
    :param int iterations: the maximum number of iterations to perform gradient descent
    :param float lambda_: the regularization parameter
    :param float learning_rate: the gradient descent learning rate
    :return: two latent factor models with the shapes M*dimensions and dimensions*N;
    :rtype: tuple 
    """
    m, n = user_movie_matrix.shape
    u = np.random.rand(m, dimensions)
    v = np.random.rand(dimensions, n)
    for iteration in xrange(iterations):
        # TODO: determine why the sqrt-decreasing learning rate results in tiny u and v values
        learning_rate_decreasing = learning_rate  # / sqrt(iteration + 1)
        for i_user in xrange(m):
            for j_movie in xrange(n):
                # Only calculate non-missing values
                if user_movie_matrix[i_user][j_movie] > 0:
                    error = user_movie_matrix[i_user][j_movie] - np.dot(u[i_user, :], v[:, j_movie])
                    # Gradient descent
                    for dimension in xrange(dimensions):
                        u[i_user][dimension] += learning_rate_decreasing * (2 * error * v[dimension][j_movie] -
                                                                            lambda_ * u[i_user][dimension])
                        v[dimension][j_movie] += learning_rate_decreasing * (2 * error * u[i_user][dimension] -
                                                                             lambda_ * v[dimension][j_movie])
        u_dot_v = np.dot(u, v)
        error_total = 0
        for i_user in xrange(m):
            for j_movie in xrange(n):
                if user_movie_matrix[i_user][j_movie] > 0:
                    error_total += (user_movie_matrix[i_user][j_movie] - u_dot_v[i_user, j_movie]) ** 2
                    # Frobenius norm
                    for dimension in xrange(dimensions):
                        error_total += lambda_ / 2 * (u[i_user][dimension] ** 2 + v[dimension][j_movie] ** 2)
        if error_total < 0.01:
            break
    return u, v


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


def test():
    test_matrix = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
        ])
    u, v = matrix_factorization(test_matrix, dimensions=2, iterations=5000, lambda_=0.02, learning_rate=0.0002)
    u_dot_v = np.dot(u, v)
    tolerance = 1
    for i in range(test_matrix.shape[0]):
        for j in range(test_matrix.shape[1]):
            rating_in = test_matrix[i][j]
            rating_out = u_dot_v[i][j]
            if rating_in:
                assert abs(rating_in - rating_out) < tolerance, 'input {} output {}'.format(rating_in, rating_out)


def run():
    user_movie_matrix = read_data(RATINGS_FILE_PATH, MOVIES_FILE_PATH)
    u, v = matrix_factorization(user_movie_matrix, dimensions=10)
    print np.dot(u, v)


if __name__ == '__main__':
    test()
    # run()
