% projection approximation method

function [w, yang_w_history, cosim_history_yang] = yang_approx(D, pc1,iter)

n = size(D,2)
w = 2.*rand(n,1)-1
D=D-mean(D,1)
for i = 1:iter
    yang_w_history{i}=w
    cosim_history_yang(i)=abs(getCosineSimilarity(pc1, w))
    w_old = w
    y_cross = sum(D*w_old.*D)
    y_sum_sq = sumsqr(D*w_old);
    w = (-y_cross/y_sum_sq)'
    w=w/norm(w)
end
end

