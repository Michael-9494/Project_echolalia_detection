function M = simmx(A,B)
% M = simmx(A,B)
%    calculate a sim matrix between specgram-like feature matrices A and B.
%    size(M) = [size(A,2) size(B,2)]; A and B have same #rows.


EA = sqrt(sum(A.^2));
EB = sqrt(sum(B.^2));

M = (A'*B)./(EA'*EB);