% Use this to generate Y-matrix from ratings matrix
y=zeros(num_users, num_movies);

u = rand(num_users, K);
v = rand(K, num_movies);
% initializing latent factors randomly

for i =1:length(ratings)
    y(ratings(i,1),ratings(i,2))= ratings(i,3);
end
%stepping through the ratings table to retrieve the ratings and store it
%under the right movie and user it came from in the y matrix

total_error=0;
iteration=1;
error=1;
while (iteration<=ITERATIONS && error>0.01)
    ita_norm=ITA/sqrt(iteration); 
    %to normalize variable on each iteration of Grad. Desc.
    error=0;
    for user = 1:num_users
        for movie = 1:num_movies
            if y(user,movie)~=0
                dEij = y(user, movie) - ((u(user,:)*v(:,movie)));
                for k = 1:K
                    u(user, k)= u(user, k)- (ita_norm * ((LAMBDA * u(user, k)) - (2 * v(k, movie) * dEij)));
                    %subtracting the derivative of the loss with respect to
                    %u
                    v(k, movie)= v(k, movie)- (ita_norm * ((LAMBDA * v(k, movie)) - (2 * u(user, k) * dEij)));
                    %subtracting the derivative of the loss with respect to
                    %u
                end
            end
        end
    end
    u_v= u*v;
    for user = 1:num_users 
        %beginning loop to measure and update error val
        for movie = 1:num_movies
            inspect=y(user, movie);
            if inspect ~= 0
                error= (inspect-u_v(user, movie))^2;
                for k= 1:K
                    error=error+ ((LAMBDA/2)* ((u(user,k))^2 + (v(k, movie))^2));
                end
            end
        end
    end
    total_error=total_error+ error;
    iteration=iteration+1;
end

u_v_printer;