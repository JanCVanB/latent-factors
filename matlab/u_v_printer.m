% outputs u and v matrix to text files

dlmwrite('out_data/u_matrix.txt',u,'delimiter','\t');
dlmwrite('out_data/v_matrix.txt',v,'delimiter','\t');
dlmwrite('out_data/u.v_matrix.txt',(u*v),'delimiter','\t');