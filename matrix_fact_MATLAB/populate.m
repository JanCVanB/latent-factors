% populate the matrix importations from files

K = 20; % inner dimensions of U.V matrixes
ITA=0.01; % Learning Rate
ITERATIONS=50; %alternative stopping condition for Gradient Descent
LAMBDA=10; % Regularization Parameter

RATINGS_FILE_PATH = 'data/data.txt';
MOVIES_FILE_PATH = 'data/movies.txt';

ratings = importdata(RATINGS_FILE_PATH);

movie_tags_raw = importdata(MOVIES_FILE_PATH);
movie_tags = movie_tags_raw.data;

num_users = length(unique(ratings(:,1)));
num_movies = length(movie_tags);