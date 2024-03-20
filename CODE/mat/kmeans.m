function centroids = kmeans(X,k)
%set error threshold
min_thresh=1e-7;
%set maximum iterations
max_iter=10000000;
%centroids
centroids = zeros(k, 2);
len = size(X,2);
nearest_c = zeros(len);
%initialise to random points
rand_i = ceil(len*rand(k, 1));
for i = 1:k
    centroids(i,:) = X(:,rand_i(i));
end
%Iteration loop
for i=1:max_iter
    %updated means
    new_c = zeros(size(centroids));
    %no, of points assigned to each mean
    assigned2c = zeros(k, 1);
    %Go through all data points
    for n=1:len
        % Calculate nearest mean
        x = X(1, n);
        y = X(2, n);
        diff = ones(k,1)*X(:,n)' - centroids;
        dist = sum(diff.^2, 2);

        [~,indx] = min(dist);
        nearest_c(n) = indx;
        new_c(indx, 1) = new_c(indx, 1) + x;
        new_c(indx, 2) = new_c(indx, 2) + y;
        assigned2c(indx) = assigned2c(indx) + 1;
    end

    %Compute new centroids
    for i = 1:k
        %Only if a centroid has data assigned
        if (assigned2c(i) > 0)
            new_c(i,:) = new_c(i,:) ./ assigned2c(i);
        end
    end

    %Early exit if error is small
    d = sum(sqrt(sum((new_c - centroids).^2, 2)));
    if d < min_thresh
        break;
    end
    centroids = new_c;
end