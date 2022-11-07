% First we set up the HMM model H, the observed sequence X and the lengths:

Pi=[0.7, 0.1, 0.2];
B=[0.1, 0.02, 0.6];
A=[0.5 0.2 0.3
    0.15 0.6 0.25
    0.1 0.4 0.5];
N=length(Pi);
X=[0 0 0 0 1 0 0];
T=length(X);
% Next we define an empty array for α(j, i) and compute the initial state:
alpha=zeros(T,N);
%initial state
alpha(1,1:N)=B(:).*Pi(:);
% Finally, iterating forwards through the observations:
for t=1:T-1
    for Pi=1:3
        alpha(t+1,Pi)=B(Pi)*sum(A(Pi,:)*alpha(t,:)');
    end
end