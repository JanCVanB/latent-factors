"""Project factorized matrices into 2 dimensions using top 2 latent factors

Last modified on April 3, 2015

.. moduleauthor:: Jan Van Bruggen <jvanbrug@caltech.edu>
"""
import csv
from itertools import cycle
import math
import matplotlib.pyplot as plt
import numpy as np


MOVIES_FILE_PATH = '../data/movies.txt'
V_FILE_PATH = 'results/after_projection/v_2dim.csv'


def graph(movies, v):
    """Graph all movies along two latent factor axes, with certain movies highlighted

    :param np.array movies: information about movies, including titles
    :param np.array v: V matrix containing response of each movie to the top 2 latent factors
    """
    xv, yv = v[0, :], v[1, :]
    specials = ['Amityville' in movie[1] for movie in movies]  # Amityville films
    # specials = [int(movie[5]) and int(movie[6]) for movie in movies]  # Children's Animations
    # specials = [any(s in movie[1] for s in ('Empire', 'Star Wars', 'Jedi')) for movie in movies]  # Star Wars films
    # specials = [bool(int(movie[12])) for movie in movies]  # Film-Noir
    # specials = [int(movie[13]) and int(movie[17]) for movie in movies]  # Horror and Sci-Fi
    # specials = [int(movie[10]) and int(movie[14]) for movie in movies]  # Drama and Musical
    # specials = [(xv[i] + 2) ** 2 + yv[i] ** 2 > 1.2 for i in range(len(movies))]  # Outer ring of films
    filter_special = lambda x: ([x[i] for i in range(len(x)) if specials[i]],
                                [x[i] for i in range(len(x)) if not specials[i]])
    movies_special, movies_other = filter_special(movies)
    xv_special, xv_other = filter_special(xv)
    yv_special, yv_other = filter_special(yv)
    plt.title('The Amityville Franchise')
    plt.xlim(-4, 0)
    plt.ylim(-2, 2)
    plt.scatter(xv_other, yv_other, facecolor='0.7', lw=0, s=7)
    plt.scatter(xv_special, yv_special, c='k', s=30)
    text_colors = cycle(['red', 'green', 'blue', 'purple', 'brown'])
    x_text_center, y_text_center = np.mean(xv_special), np.mean(yv_special)
    x_text_radius = max([xv - x_text_center for xv in xv_special])
    y_text_radius = max([yv - y_text_center for yv in yv_special])
    for index, movie in enumerate(movies_special):
        angle = math.atan2(yv_special[index] - y_text_center, xv_special[index] - x_text_center)
        x_text = 2 * x_text_radius * math.cos(angle) + x_text_center
        y_text = 2 * y_text_radius * math.sin(angle) + y_text_center
        color = next(text_colors)
        plt.annotate(movie[1], xy=(xv_special[index], yv_special[index]),
                     xytext=(x_text, y_text),
                     arrowprops=dict(arrowstyle='->', ec=color), ha='center', color=color, size=12)
    plt.show()


def read_data():
    """Read data from U, V matrices
    """
    with open(MOVIES_FILE_PATH, 'rU') as movie_file:
        movies = np.array(list(csv.reader(movie_file, dialect=csv.excel_tab)))
    v = np.genfromtxt(V_FILE_PATH, delimiter=',')
    return movies, v


def run():
    movies, v = read_data()
    graph(movies, v)


if __name__ == '__main__':
    run()
