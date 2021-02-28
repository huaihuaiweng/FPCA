%calculate the explained variance of the given PC

function explained_var = ev(w, D)

% explained_var = norm(D*w)^2/sum(diag(cov(D)))
explained_var = var(D*w)/sum(diag(cov(D)))
