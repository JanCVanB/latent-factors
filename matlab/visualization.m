U_FILE_PATH = 'data/python_data/u.csv';
V_FILE_PATH = 'data/python_data/v.csv';

u_c= importdata(U_FILE_PATH);
v_c= importdata(V_FILE_PATH);

%[A_u, E_u, B_u] = svd (transpose(u_c));
[A_v, E_v, B_v] = svd (v_c);

%U=transpose(A_v(:,1:2))*transpose((u_c));

V=transpose(A_v(:,1:2))*v_c;

hold off;
scatter(V(1,:),V(2,:));
hold on;
scatter(V(1,437:442), V(2,437:442)) % specifically highlighting Amityville sequel here
c = movie_tags_raw.textdata(437:442,2);

dy = 0.02; % displacement so the text does not overlay the data points

text(V(1,437:442), V(2,437:442)+dy, c); %specifically selecting amitt