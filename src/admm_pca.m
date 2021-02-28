function [z, time, pc1_cosin_history, ev_history, w_history, z_history, r_norm, s_norm, t] = admm_pca(K, D, data_mat, pc1, rho, iter)
% INPUT
% K: number of workers
% D: worker data
% data_mat: centered dataset
% pc1: the first PC attained by SVD method 
% rho: the penalty coefficent for ADMM
% iter: number of iterations for ADMM

% OUTPUT
% z: the final model
% time: the running time
% pc1_cosin_history: record of cosine similarity with pc1 for each iteration
% ev_history: the record of explained variance for each iteration
% r_norm: the record of primal residual
% s_norm: the record of dual residual
% t: record of t-th iteration at which the training stops

% rand('seed', 1);
n = size(D{1},2); %dimension of the data
w = rand(n,K);%record of the model of each worker
z = zeros(n,1)
u = zeros(n,K);%record of the dual variable of each worker
epsilon_abs = 1e-3; %parameter for stopping criterion
epsilon_rel = 1e-3; %parameter for stopping criterion

pc1_cosin_history=0; %initialization
loss=0;%initialization
flag=0;%flag for signaling break

tic;
for t = 1:iter
	% record updated w
    w_history{t}=w;

    % w from the last iteration
    w_old = w;

    % for each worker do
    for i = 1:K
        y_cross = 0;
        y_sum_sq = 0;
        y = D{i}*w_old(:,i);
        y_cross = 2*D{i}'*y;
        y_sum_sq = 2*y'*y;
        w_tmp = (y_cross+rho*z-u(:,i)) / (y_sum_sq + rho); %the updated w
        w(:,i) = w_tmp / norm(w_tmp); %normalize the updated w 
    end
    for i =1:K
        cosim=getCosineSimilarity(w(:,1),w(:,i));
        if (cosim<0)
            w(:,i) = -w(:,i);
        end
    end
    
    %master update
    w_ave = mean(w,2); 
    z_history{t} = z;
    z_old = z; %the z from last iteration (t-1)
    z_tmp = w_ave+1/rho*(mean(u,2));
    z = z_tmp/norm(z_tmp);
    u = u+rho*(w-z);
    r_norm(t)  = max(vecnorm(w-z));
    s_norm(t)  = norm(rho*(z-z_old));
    
    %varying rho
    if (r_norm(t) > 10*s_norm(t))
        rho=rho*2
    elseif (s_norm(t) > 10*r_norm(t))
        rho=rho*1/2
    else
        rho=rho
    end
    
    %stopping criterion
    epsilon_pri = sqrt(n)*(epsilon_abs)+(epsilon_rel)*max(norm(w), norm(z));
%     epsilon_dual = sqrt(n)*(epsilon_abs)+(epsilon_rel)*norm(rho*u);
    epsilon_dual = sqrt(n)*(epsilon_abs)+(epsilon_rel)*norm(u)
        if (r_norm(t) < epsilon_pri && s_norm(t) < epsilon_dual)
            flag=1;
            break;
        end
    
    %record cosine similarity with the first PC attained by SVD method
    pc1_cosin_history(t)=abs(getCosineSimilarity(pc1,z));
    
    %record explained variance
    ev_history(t)=ev(z,data_mat);
    
    %record MSE 
    loss(t)=norm(data_mat.'-z*z.'*data_mat.');
    
    if (flag == 1)
        break;
    end
end
time = toc;
