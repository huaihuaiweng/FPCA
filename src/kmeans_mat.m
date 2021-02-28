% set the opt to '0' since we no longer use kmeans_opt 

function [reduced_set] = kmeans_mat(data, worker_nodes, num_cluster, opt)
%opt == 0 -> use kmeans
%opt == 1 -> use kmeans_opt
reduced_set = []
K = size(worker_nodes,2)
if (strcmpi (opt,'0'))
    for i = 1:K
        b = sum(worker_nodes(1:i))
        if i == 1
            [IDX{i},C{i},SUMD{i},numK{i}] = kmeans(data(1:b,:), num_cluster)
            reduced_set = [reduced_set;C{i}]
        else
            [IDX{i},C{i},SUMD{i},numK{i}] = kmeans(data(sum(worker_nodes(1:i-1))+1:b,:), num_cluster)
            reduced_set = [reduced_set;C{i}]
        end
    end
    
elseif (strcmpi (opt,'1'))
    for i = 1:K
    b = sum(worker_nodes(1:i))
    if i == 1
        [IDX{i},C{i},SUMD{i},numK{i}] = kmeans_opt(data(1:b,:))
        reduced_set = [reduced_set;C{i}]
    else
        [IDX{i},C{i},SUMD{i},numK{i}] = kmeans_opt(data(sum(worker_nodes(1:i-1))+1:b,:))
        reduced_set = [reduced_set;C{i}]
    end
end


end

end

