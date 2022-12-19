function [Filter_Bank,center_Frequencies,MelFrequencyVector,BW,M_tilda,Filter_Bank_of_ones]...
    = Mel_Filter_bank(range,WindowLenSamp,Fs,NumBands)


mel_range = hz2mel(range);
bandEdges = mel2hz(linspace(mel_range(1),mel_range(end),NumBands+2));
center_Frequencies = bandEdges(2:end-1);

linFq = (0:WindowLenSamp-1)/WindowLenSamp*Fs;
dif_freq = 1/WindowLenSamp*Fs;
MelFrequencyVector = 2595*log10(1+linFq/700); % phy(f)
% Determine inflection points
filterBandWidth = bandEdges(3:end) - bandEdges(1:end-2);
weightPerBand   = filterBandWidth./2;

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

Linear_ind = index-1  ;  % sample numbers in the linear  domain
kk = 2595*log10(1+(((Fs/2)*linFq)/Linear_ind)/700)*(NumBands/(Fs/2));%MelFrequencyVector
round_kk = round(kk);

M_tilda = zeros(NumBands,Linear_ind);
Filter_Bank = zeros(NumBands,Linear_ind)';%Mel-mapping matrix
Filter_Bank_of_ones = zeros(NumBands,Linear_ind);

for k = 1:numel(bandEdges)-2
    % Rising side of triangle
    for j = p(k):p(k+1)-1
        Filter_Bank(j,k) = ...
            ((linFq(j) - bandEdges(k)) / bw(k))./weightPerBand(k);

        if abs(bandEdges(k+1)-linFq(j))<dif_freq/2
            Filter_Bank_of_ones(k,j) = 1;
            M_tilda(k,j) = Filter_Bank(j,k);
        end

    end

    % Falling side of triangle
    for j = p(k+1):p(k+2)-1
        Filter_Bank(j,k) = ...
            (bandEdges(k+2) - linFq(j)) / bw(k+1)./weightPerBand(k);

        if abs(bandEdges(k+1)-linFq(j))<dif_freq/2
            Filter_Bank_of_ones(k,j) = 1;
            M_tilda(k,j) = Filter_Bank(j,k);
        end

    end


end


% The weights of each bandpass filter are normalized by the corresponding
% bandwidth of the filter
% for i = 1:NumBands
%     Filter_Bank(:,i) = Filter_Bank(:,i)./weightPerBand(i);
% end
Filter_Bank = Filter_Bank';
end