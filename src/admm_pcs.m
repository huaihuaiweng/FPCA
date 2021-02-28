%find PCs derived from admm_pca.m

function [pc_list,pc_cosim_list, z_list, primal_history, dual_history] = admm_pcs(num_pc, data, D, rho_ls, coeffs, iter)
K = size(D,2)

for i = 1:num_pc
    best_rho=rho_ls(i)
    [z, time, pc1_cosin_history, ev_history, w_history, z_history,r_norm, s_norm, t] = admm_pca(K, D, data,coeffs(:,i), best_rho, iter)
    pc_list(:,i) = z
    pc_cosim_list{i} = pc1_cosin_history
    z_list{i} = z_history
    primal_history{i} = r_norm
    dual_history{i} = s_norm
    for j = 1:K
        D{j} = D{j} - D{j}*z*z'
    end
end
    data = cell2mat(D')
end

