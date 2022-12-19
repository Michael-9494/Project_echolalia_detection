function pdf = gaussian(x, mu, sigma)

for n=1:size(sigma,1)/size(mu,1)
    diff(:,n)=x(:,n)-mu(n);
end %ends with n set to the size difference
pdf = 1 / sqrt((2*pi)^n * det(sigma))* exp(-1/2 * sum((diff*inv(sigma) .* diff), 2));
end