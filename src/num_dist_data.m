function [ratio_ls] = num_dist_data(n,N)
% rand('seed', 1);
ratio_ls=0;
init = N;
for i=1:n
    if i == n
        ratio_ls(i)=N
        break
    end
    min=floor(init/(n+1))
    max=floor(init/(n))
    share=randi([min max],1)
    N=N-share
    ratio_ls(i)=share
end
end