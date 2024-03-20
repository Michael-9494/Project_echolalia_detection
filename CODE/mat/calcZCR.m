function ZeroCrossingSignal = calcZCR(framedSignal)
% calcNRG calculate the the energy of each frame
% framedSignal – a matrix of the framed signal, after preprocessing
% OUTPUT:
% ZeroCrossingSignal – a column vector of the zero-crossing values of the signal
[M,N]=size(framedSignal);
% framedSignal = framedSignal-mean(framedSignal,2);

framedSignal1 = framedSignal;
framedSignal2 = cat(2,zeros(M,1),framedSignal(:,1:end-1));
ZeroCrossingSignal = (1/(2*(M*N-1)))*sum( abs( sgn( framedSignal1 )-...
                            sgn( framedSignal2) ),2 );

end