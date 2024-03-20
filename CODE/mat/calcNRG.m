function EnergySignal=calcNRG(framedSignal)
% calcNRG calculate the energy of each frame
% framedSignal – a matrix of the framed signal, after preprocessing
% OUTPUT:
% EnergySignal – a column vector of the energy values of the signal

% take the length of each frame
[~,N]=size(framedSignal); 
% sum( _ ,2) in order to sum each frame
EnergySignal=20*log10((1/N)*(sum((framedSignal).^2,2)));

end
