function [p,q,C] = dp(A)
% [p,q] = dp(M)
%    Use dynamic programming to find a min-cost path through matrix M.
%    Return state sequence in p,q
% 2003-03-15 dpwe@ee.columbia.edu

% Copyright (c) 2003 Dan Ellis <dpwe@ee.columbia.edu>
% released under GPL - see file COPYRIGHT

[N,M] = size(A);

% costs
C = zeros(N+1, M+1);
C(1,:) = NaN;
C(:,1) = NaN;
C(1,1) = 0;
C(2:(N+1), 2:(M+1)) = A;

% traceback
phi = zeros(N,M);

for i = 1:N
    for j = 1:M
        [dmax, tb] = min([C(i, j), C(i, j+1), C(i+1, j)]);
        C(i+1,j+1) = C(i+1,j+1)+dmax;
        phi(i,j) = tb;
    end
end

% Traceback from top left
i = N;
j = M;
p = i;
q = j;
while i > 1 && j > 1
    tb = phi(i,j);
    if (tb == 1) %match
        i = i-1; j = j-1;
    elseif (tb == 2) %insertion
        i = i-1;
    elseif (tb == 3) %deletion
        j = j-1;
      else
          error;
    end
    p = [i,p];%Rows
    q = [j,q];%Columns
end

% Strip off the edges of the D matrix before returning
C = C(2:(N+1),2:(M+1));