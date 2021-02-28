clear; 
for time = 1:10
    %data preprocessing
    %------------------
    data = importdata('synthetic_control.data')
    y = []
    for i=1:6
        tmp = repelem([i], 100)
        y = [y tmp]
    end
    y=y'

    data=[data y]
    shuffle_idx = randperm(size(data,1))
    data = data(shuffle_idx,:)
    y = data(:,61)
    data = data(:,1:60)
    %------------------
    
    m = size(data,1) %whole datasize
    K = 5 %numebr of workers
    worker_size_ls = num_dist_data(K, m) %decide distributed workers K:number of workers; m:dataset size

    reduced_set = kmeans_mat(data, worker_size_ls, 6, '0')
    kernel_mat = KGaussian(1/50^2, data, reduced_set)
    centered_data = kernel_mat-mean(kernel_mat)
    [coeff,score,latent,tsquared,explained,mu] = pca(kernel_mat)

    D = distr_data(worker_size_ls, centered_data)
    num_pcs = 2
    rho = repelem([1], num_pcs)
    iter = 1000;
    [pc_list,pc_cosin_history, z_list, primal_history, dual_history] = admm_pcs(num_pcs, kernel_mat, D, rho, coeff, iter)
    [yang_pc_list, yang_cosin_history, yang_z_list] = yang_pcs(num_pcs, kernel_mat, coeff, 100)
    
    pc_cosin_history_10{time} = pc_cosin_history
    yang_cosin_history_10{time} = yang_cosin_history
    
    primal_history_10{time} = primal_history
    dual_history_10{time} = dual_history

end

exp_name = 'synthetic'
pc = 2
iter_ls = 0
for time = 1:10
    for iter = 1:size(pc_cosin_history_10{time}{pc},2)
        if (pc_cosin_history_10{time}{pc}(iter)>=0.9999)
            iter_ls(time) = iter
            break
        end
    end
end
max(iter_ls)

hold on
for time = 1:10
    plot(pc_cosin_history_10{time}{pc}(1:end))
end
hold off
xline(max(iter_ls),'--')
title(['PC', num2str(pc), ' Cosine Similarity Convergence plot'])
xlabel('iterations')
ylabel('Cosine Similarity')
xlim([1 30])
ylim([0 1])
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_pc', num2str(pc),'.png'])

pc = 2
iter_ls = 0
for time = 1:10
    for iter = 1:size(yang_cosin_history_10{time}{pc},2)
        if (yang_cosin_history_10{time}{pc}(iter)>=0.9999)
            iter_ls(time) = iter
            break
        end
    end
end
max(iter_ls)
%yang
hold on
for time = 1:10
    plot(yang_cosin_history_10{time}{pc}(2:20))
end
hold off
xline(max(iter_ls),'--')
title(['PC', num2str(pc),' Cosine Similarity Convergence plot'])
xlabel('iterations')
ylabel('Cosine Similarity')
ylim([0 1])
xlim([1 10])
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_yangpc', num2str(pc), '.png'])


%primal  residaul 
hold on
for time = 1:10
    plot(primal_history_10{time}{pc}(1:end))
end
hold off
xlim([1 200])
title(['PC', num2str(pc), ' Primal Residual Convergence Plot'])
xlabel('iterations')
ylabel('Primal Residual')
% ylim([0 10])
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_pc', num2str(pc),'_prires.png'])

%dual
hold on
for time = 1:10
    plot(dual_history_10{time}{pc}(1:end))
end
hold off
xlim([1 200])
title(['PC', num2str(pc), ' Dual Residual Convergence Plot'])
xlabel('iterations')
ylabel('Dual Residual')
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_pc', num2str(pc),'_dualres.png'])


save('exp_synthetic_10_0525')
load('exp_synthetic_10_0525')