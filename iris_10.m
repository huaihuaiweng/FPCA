clear;
for time = 1:10
    %data preprocessing
    %------------------
    disp(time)
    FID = fopen('iris.data')
    C_data0 = textscan(FID,'%f %f %f %f %s', 200, 'Delimiter',',')
    data = cell2mat(C_data0(:,1:4)) %ignores the last column of strings

    iter=1000;
    m = size(data,1) %whole datasize
    K = 5 %numebr of workers
    worker_size_ls=num_dist_data(K, m) %decide distributed workers K:number of workers; m:dataset size
    data=data-mean(data)
    D = distr_data(worker_size_ls, data)
    [coeff,score,latent,tsquared,explained,mu]=pca(data)

    y=0
    for i=1:size(data,1)
        tmp=string(C_data0{5}(i))
        if tmp=='Iris-setosa'
            y(i)=1
        elseif tmp=='Iris-versicolor'
            y(i)=2
        else
            y(i)=3
        end
    end
    data(:,5)=y'
    %------------------
    
    num_pcs = 3 % set numbers of pc to compute
    rho = repelem([1], num_pcs) %initialize rho
    [pc_list,pc_cosin_history, z_list, primal_history, dual_history] = admm_pcs(num_pcs, data(:,1:4), D, rho, coeff, iter)
    [yang_pc_list, yang_cosin_history, yang_z_list] = yang_pcs(num_pcs, data(:,1:4), coeff, 100)
    
    pc_cosin_history_10{time} = pc_cosin_history
    yang_cosin_history_10{time} = yang_cosin_history
    
    primal_history_10{time} = primal_history
    dual_history_10{time} = dual_history
end


% draw stopping line(similarity >= 0.9999 )
iter_ls = 0
for time = 1:10
    for iter = 1:size(pc_cosin_history_10{time}{pc},2)
        if (pc_cosin_history_10{time}{pc}(iter)>=0.9999)
            iter_ls(time) = iter
            break
        end
    end
end
%find the index of the stopping line
max(iter_ls)

%plot 10 times experiments of FPCA
exp_name = 'iris'
pc = 1
hold on
for time = 1:10
    plot(pc_cosin_history_10{time}{pc}(1:end))
end
hold off
title(['PC', num2str(pc), ' Cosine Similarity Convergence plot'])
xlabel('iterations')
xline(max(iter_ls),'--')
xlim([1 20])
ylabel('Cosine Similarity')
ylim([0.94 1])
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_pc', num2str(pc),'.png'])


% find stopping line 
for time = 1:10
    for iter = 1:size(yang_cosin_history_10{time}{pc},2)
        if (yang_cosin_history_10{time}{pc}(iter)>=0.9999)
            iter_ls(time) = iter
            break
        end
    end
end
max(iter_ls)

%plot 10 times experiments of projection approximatation method
pc = 2
iter_ls = 0

hold on
for time = 1:10
    plot(yang_cosin_history_10{time}{pc}(2:11))
end
hold off
xline(max(iter_ls),'--')
title(['PC', num2str(pc),' Cosine Similarity Convergence plot'])
xlabel('iterations')
ylabel('Cosine Similarity')
xlim([1 10])
ylim([0 1])
saveas(gcf,['/Users/weng/Documents/pca_plot2/',exp_name,'_10_yangpc', num2str(pc), '.png'])


%primal  residaul 
hold on
for time = 1:10
    plot(primal_history_10{time}{2}(1:16))
end
hold off
xlim([1 16])
title('PC2 Primal Residual Convergence Plot')
xlabel('iterations')
ylabel('Primal Residual')
saveas(gcf,['/Users/weng/Documents/pca_plot2/iris_10_pc2_prires.png'])


%dual residual 
hold on
for time = 1:10
    plot(dual_history_10{time}{2}(1:16))
end
hold off
xlim([1 16])
title('PC2 Dual Residual Convergence Plot')
xlabel('iterations')
ylabel('Dual Residual')
saveas(gcf,['/Users/weng/Documents/pca_plot2/iris_10_pc2_dualres.png'])




