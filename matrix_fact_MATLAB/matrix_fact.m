% Use this to generate Y-matrix from ratings matrix
y=zeros(num_users, num_movies);

u = rand(num_users, K);
v = rand(K, num_movies);

for i =1:length(ratings)
    y(ratings(i,1),ratings(i,2))= ratings(i,3);
end

iteration=1;
error=1;
while (iteration<=ITERATIONS && error>0.01)
    ita_norm=ITA/sqrt(iteration); %to normalize variable on each iteration of Grad. Desc.
    error=0;
    for user = 1:num_users
        for movie = 1:num_movies
            if y(user,movie)~=0
                dEij = y(user, movie) - (u(user,:)*v(:,movie));
                for k = 1:K
                    u(user, k)= u(user, k)- ita_norm * (LAMBDA * u(user, k) - 2 * v(k, movie) * dEij);
                    v(k, movie)= v(k, movie)- ita_norm * (LAMBDA * v(k, movie) - 2 * u(user, k) * dEij);
                end
            end
        end
    end
    u_v= u*v;
    for user = 1:num_users
        for movie = 1:num_movies
            inspect=y(user, movie);
            if inspect ~= 0
                error= (inspect-u_v(user, movie))^2;
                for k= 1:K
                    error=error+ ((LAMBDA/2)* (u(user,k)^2 + v(k, movie)^2));
                end
            end
        end
    end
    iteration=iteration+1;
end