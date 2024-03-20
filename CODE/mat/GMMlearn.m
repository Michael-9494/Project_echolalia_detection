function [mu,sigma]=GMMlearn(X,k)

% Just as with k-means, GMMs begin with an initial estimate, and then iteratively assign
% data points to mixture components before re-estimating the mixture components based
% on the data point allocation, working towards a locally optimum solution.
% In this case we have K Gaussians, Nk, each defined by their mean, μk, and covariance
% matrix, k:

%set error threshold
min_thresh=1e-12;
%set maximum iterations
max_iter=1000;
% number of data points.
m = size(X, 1);
% dimension
n = size(X, 2);
%same initial probabilities
phi = ones(1, k) * (1 / k);

%random data points as initial mu's
indeces = randperm(m);
mu = X(indeces(1:k), :);
sigma = [];

%cov of all data as initial sigma's
for (j = 1 : k)
    sigma{j} = cov(X);
end

%initial probabilities
w = zeros(m, k);

for (iter = 1:max_iter)
    pdf = zeros(m, k);

    %for k mixtures, prob of data point per mixture
    for (j = 1 :k)
        pdf(:, j) = gaussian(X, mu(j, :), sigma{j});
    end
    %calculate pdf_w
    pdf_w=pdf.*repmat(phi,m,1);
    %calculate w
    w=pdf_w./repmat(sum(pdf_w, 2),1,k);

    %store previous means
    old_mu = mu;

    %for k mixtures
    for (j = 1 : k)
        %prior prob. for component j
        phi(j) = mean(w(:, j), 1);
        %new mean for component j
        mu(j, :) = (w(:, j)'*X)./sum(w(:, j),1);
        %new cov matrix (for current component)
        sigma_k = zeros(n, n);
        %subtract cluster mean from all data points
        Xm = bsxfun(@minus, X, mu(j, :));
        %contrib of each data vector to the cov matrix
        for (i = 1 : m)
            sigma_k = sigma_k + (w(i, j) .* (Xm(i, :)'* Xm(i, :)));
        end
        %normalise
        sigma{j} = sigma_k ./ sum(w(:, j));
    end
    %early termination on convergence
    if abs(mu - old_mu) < min_thresh
        break
    end
end