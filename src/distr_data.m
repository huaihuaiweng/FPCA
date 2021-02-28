function [D] = distr_data(worker_nodes,data)
% allocate the data to workers
%center the data matrix and allocate the data to workers
K=size(worker_nodes,2)

for i = 1:K
    b = sum(worker_nodes(1:i))
    if i==1
        D{i}=data(1:b,:)
    else
        D{i}=data(sum(worker_nodes(1:i-1))+1:b,:)
    end
end

end
