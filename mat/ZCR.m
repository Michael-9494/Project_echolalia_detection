function ZCR = ZCR(x)
% calcNRG calculate the the energy of each frame
% framedSignal – a matrix of the framed signal, after preprocessing
% OUTPUT:
% ZeroCrossingSignal – a column vector of the zero-crossing values of the signal

% framedSignal = framedSignal-mean(framedSignal,2);
n = size(x);
ZCR = zeros([1 n(1)]);
for j = 1:n

ZCR(j) = length(find(diff(cumsum(sign(x(j,:))))==-1));
end

end