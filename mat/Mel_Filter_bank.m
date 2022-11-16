function [Filter_Bank,center_Frequencies,MelFrequencyVector,BW] = Mel_Filter_bank(range,WindowLenSamp,Fs,NumBands)
mel_range = hz2mel(range);
bandEdges = mel2hz(linspace(mel_range(1),mel_range(end),NumBands+2));
center_Frequencies = bandEdges(2:end-1);

linFq = (0:WindowLenSamp-1)/WindowLenSamp*Fs;
MelFrequencyVector = 2595*log10(1+linFq/700);
% Determine inflection points

p =  zeros(numel(bandEdges),1);

for edgeNumber = 1:numel(bandEdges)
    for index = 1:length(linFq)
        if linFq(index) > bandEdges(edgeNumber)
            p(edgeNumber) = index;
            break;
        end
    end
end
BW = bandEdges(3:end) - bandEdges(1:end-2);
bw = diff(bandEdges);
for k = 1:numel(bandEdges)-2
    % Note regarding the "for j = .."  loops below.
    % They are needed for codegen support. Rewriting them
    % as matrix index expressions would cause this function to
    % fail constant folding and thus codegen won't allow
    % to use this function to compute a nontunable property.
    %
    % Rising side of triangle
    for j = p(k):p(k+1)-1
        Filter_Bank(j,k) = ...
            (linFq(j) - bandEdges(k)) / bw(k);
    end

    % Falling side of triangle
    for j = p(k+1):p(k+2)-1
        Filter_Bank(j,k) = ...
            (bandEdges(k+2) - linFq(j)) / bw(k+1);
    end


end

filterBandWidth = bandEdges(3:end) - bandEdges(1:end-2);
weightPerBand   = filterBandWidth./2;
% The weights of each bandpass filter are normalized by the corresponding
% bandwidth of the filter
for i = 1:NumBands
    Filter_Bank(:,i) = Filter_Bank(:,i)./weightPerBand(i);
end
Filter_Bank = Filter_Bank';
end