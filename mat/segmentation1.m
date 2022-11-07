function [segment_ind,delta]=segmentation1(signal,winlen,eta,dt,Fs,Idx)
% this funcrion takes speech signal and convert it to segments based on
% spectral error measure
% signal – the speech signal after preprocessing
% winlen – length of test and reference windows [seconds]
% eta – threshold for Δ1 spectral error measure
% dt – minimum time above threshold 'eta' [seconds]
% Fs – sampling frequency
% Idx- start & end indices of the word [samples]
% OUTPUT
% Seg_ind – index for the beginning of each segment
% delta – spectral error measure Δ1


% stay only with the indices of the word
delta=zeros(length(signal),1);
% Num in samples window reference, test & dt [S]*[sample/S]=[sample]:
N = winlen*Fs;
dt_in_samples = dt*Fs;
% start the problem from Idx(1)
index_ref=Idx(1);
index_test=Idx(1);
segment_ind=Idx(1);

% main while loop.. end when the next reference window + dt in samples will
% cross the length of Idx(2)
while index_ref+N-1+dt_in_samples <= Idx(2)
    flag=1;
    % create reference and test signals
    Ref=signal(index_ref:index_ref+N-1);
    % another loop for the test window. stop when the end of test will cross
    % Idx(2) or when delta>eta for at least dt time
    while   index_test+N-1<= Idx(2) && flag==1
        Test=signal(index_test:index_test+N-1);
        % assuming  we have a wide-sense stationary random process.
        % calculate the periodogram of each window
        pxx_ref_win=periodogram(Ref,[],length(Ref));
        pxx_test_win=periodogram(Test,[],length(Ref));
        % create omega vector to integrate ovet it. the signal is real so
        % we will intagrate over [-pi,pi]
        w=linspace(-pi,pi,length(pxx_ref_win));
        % calculate spectral error measure over test and ref window
        delta_numerator=(2*pi)*trapz(w,((pxx_test_win-pxx_ref_win).^2));
        Delta=(delta_numerator)*(1/( trapz(w,((pxx_test_win)))*...
            trapz(w,((pxx_ref_win))) ));
        % update the spectral error measure Δ1 in delta vector
%         up_idx = index_test;
        delta(index_test)=Delta;
        % verify that we reached threshold and that the minimum length of
        % foneme is 20 [ms]
        if Delta>eta && index_test-index_ref >((110*10^-3)*Fs) 
            % verify that that the threshold is reached for dt time
            How_many_delta_greater_then_eta = delta(index_test-dt_in_samples:index_test)>eta;
            check_dt=sum(How_many_delta_greater_then_eta);
            % if we did, update flag for us to go for the next segment
            if check_dt>dt_in_samples && index_test-index_ref >((110*10^-3)*Fs)
                flag=0; % go to next segment  && index_test-index_ref >((100*10^-3)*Fs)

            end
        end

        % move the test window index by one
        index_test=index_test+1;
    end
    % update the segment indices, add one idex on the rigth side of segment_ind
    if check_dt>=dt_in_samples

        segment_ind=[segment_ind (index_test-(dt_in_samples))];
    end
    % update the test and reference indices before advancing to the
    % next segment:
    % for the index_test: calculate where we started to cross the threshold
    % with the test window. and then subtract the length of dt in [samples]
    index_ref=index_test-(dt_in_samples);
    % now we can update
    index_test=index_ref;
end


% give the final delata as the current last one in order delta vec will be
% at the length of the signal
delta(index_test:end)=0;
% the last segment is the Idx(2)
segment_ind=[segment_ind Idx(2)];

end
