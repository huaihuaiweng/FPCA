clear;
for time = 1:10
    %data preprocessing
    %------------------
    data = importdata('breast_cancer.csv')
    data = data.data
    shuffle_idx = randperm(size(data,1))
    data = data(shuffle_idx,:)
    y = data(:,32)
    data = data(:,2:31)
    data = normalize(data)
    %------------------

    m = size(data,1) %whole datasize
    K = 5 %numebr of workers

    worker_size_ls = num_dist_data(K, m) %decide distributed workers K:number of workers; m:dataset size
    centered_data = data - mean(data)
    D = distr_data(worker_size_ls, centered_data)

    [coeff,score,latent,tsquared,explained,mu] = pca(data)

    num_pcs = 2
    rho = repelem([1], num_pcs)

    iter = 1000;
    [pc_list,pc_cosin_history, z_list, primal_history, dual_history] = admm_pcs(num_pcs, data, D, rho, coeff, iter)
    [yang_pc_list, yang_cosin_history, yang_z_list] = yang_pcs(num_pcs, data, coeff, 100)
    
    pc_cosin_history_10{time} = pc_cosin_history
    yang_cosin_history_10{time} = yang_cosin_history
    
    primal_history_10{time} = primal_history
    dual_history_10{time} = dual_history

end

exp_name ='breast'
pc = 1
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
title(['PC', num2str(pc), ' Cosine Similarity Convergence plot'])
xlabel('iterations')
xline(max(iter_ls),'--')
xlim([1 80])
ylabel('Cosine Similarity')
ylim([0 1])
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_pc', num2str(pc),'.png'])

pc=2
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
hold on
for time = 1:10
    plot(yang_cosin_history_10{time}{pc}(2:20))
end
hold off
xline(max(iter_ls),'--')
title(['PC', num2str(pc),' Cosine Similarity Convergence plot'])
xlabel('iterations')
ylabel('Cosine Similarity')
xlim([1 12])
ylim([0 1])
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


save('exp_breast_10_0525')
load('exp_breast_10_0525')