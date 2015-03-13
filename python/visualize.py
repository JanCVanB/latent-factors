import csv
from itertools import cycle
import math
import matplotlib.pyplot as plt
import numpy as np
from random import random


MOVIES_FILE_PATH = '../data/movies.txt'
U_FILE_PATH = 'results/after_svd/u_2dim.csv'
V_FILE_PATH = 'results/after_svd/v_2dim.csv'


def graph(movies, u, v):
    xu, yu = u[0, :], u[1, :]
    xv, yv = v[0, :], v[1, :]
    # specials = [int(movie[5]) and int(movie[6]) for movie in movies]  # Children's Animations
    specials = ['Amityville' in movie[1] for movie in movies]  # Amityville films
    # specials = [any(s in movie[1] for s in ('Empire', 'Star Wars', 'Jedi')) for movie in movies]  # Star Wars films
    # specials = [bool(int(movie[12])) for movie in movies]  # Film-Noir
    # specials = [int(movie[13]) and int(movie[17]) for movie in movies]  # Horror and Sci-Fi
    # specials = [int(movie[10]) and int(movie[14]) for movie in movies]  # Drama and Musical
    # specials = [(xv[i] + 2) ** 2 + yv[i] ** 2 > 1.2 for i in range(len(movies))]  # Outer ring of films
    # specials = [abs(yv[i]) > 1 for i in range(len(movies))]
    filter_special = lambda x: ([x[i] for i in range(len(x)) if specials[i]],
                                [x[i] for i in range(len(x)) if not specials[i]])
    movies_special, movies_other = filter_special(movies)
    xv_special, xv_other = filter_special(xv)
    yv_special, yv_other = filter_special(yv)
    # plt.title('The Star Wars Franchise')
    plt.xlim(-4, 0)
    plt.ylim(-2, 2)
    plt.scatter(xv_other, yv_other, facecolor='0.7', lw=0, s=7)
    plt.scatter(xv_special, yv_special, c='k', s=30)
    text_colors = cycle(['red', 'green', 'blue', 'purple', 'brown'])
    x_text_center, y_text_center = np.mean(xv_special), np.mean(yv_special)
    x_text_radius = max([x - x_text_center for x in xv_special])
    y_text_radius = max([y - y_text_center for y in yv_special])
    for i, movie in enumerate(movies_special):
        angle = math.atan2(yv_special[i] - y_text_center, xv_special[i] - x_text_center)
        x_text = 2 * x_text_radius * math.cos(angle) + x_text_center
        y_text = 2 * y_text_radius * math.sin(angle) + y_text_center
        color = next(text_colors)
        plt.annotate(movie[1], xy=(xv_special[i], yv_special[i]),
                     xytext=(x_text, y_text),
                     arrowprops=dict(arrowstyle='->', ec=color), ha='center', color=color, size=12)
    plt.show()


def read_data():
    """ Read data from u, v
    """
    with open(MOVIES_FILE_PATH, 'rU') as movie_file:
        movies = np.array(list(csv.reader(movie_file, dialect=csv.excel_tab)))
    u = np.genfromtxt(U_FILE_PATH, delimiter=',')
    v = np.genfromtxt(V_FILE_PATH, delimiter=',')
    return movies, u, v


def run():
    movies, u, v = read_data()
    graph(movies, u, v)


if __name__ == '__main__':
    run()
