% find PCs derived from the projection approximation  method
function [pc_list, yang_cosin_history, yang_z_list] = yang_pcs(num_pc, data, coeff, iter)
n = size(data,2)
w = rand(n,1)
    for i=1:num_pc
        [w, yang_w_history, cosim_history_yang] = yang_approx(data, coeff(:,i), iter)
        pc_list(:,i)=w
        yang_cosin_history{i}=cosim_history_yang
        yang_z_list{i}=yang_w_history
        data = data - data*w*w'
    end
end

