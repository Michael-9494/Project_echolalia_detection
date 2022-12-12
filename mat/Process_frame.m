function [frames_out] = Process_frame(frames,Fs,Param,flag,VTLN,alpha)
%PROCESS_FRAME Summary of this function goes here
%   Detailed explanation goes here
% select_gpu("auto")
coder.gpu.kernelfun
%load the struct containing the speech frames
segments = size(frames);
frames_out = struct();
p = Fs/1000 +2;
for i = 1:segments(2)
    % load each speech segment with its labels
    if isempty(frames(i).data)
        continue
    else
        % Y = fft(St);                    % perform fft > it gives double-spectrum
        % P2 = (Y/nt);                    % distribute energy
        % P1 = P2(1:nt/2+1);              % get one-side spectrum
        % P1(2:end-1) = 2*P1(2:end-1);    % Multiple by 2 as a correction for amplitude
        % Sf = P1;
        frames_out(i).Signal_frame = frames(i).data;
        frames_out(i).segment_speaker = frames(i).speakreLabel;
        frames_out(i).segment_event = frames(i).event;
        %         time2freq
        P2 = fft(frames_out(i).Signal_frame)/length(frames_out(i).Signal_frame);% perform fft > it gives double-spectrum and distribute energy
        freqs = P2(1:floor(length(frames_out(i).Signal_frame)/2)+1);              % get one-side spectrum
        freqs(2:end-1) = 2*freqs(2:end-1);    % Multiple by 2 as a correction for amplitude
        
        [~,FramedSig] = PreProcess(...
            frames_out(i).Signal_frame,Fs,Param.alpha,Param.WindowLength,Param.Overlap);
        
        T = 1/Fs;             % Sampling period
        Len = length(frames_out(i).Signal_frame);
        
        frames_out(i).t = frames(i).start_time + (0:Len-1)*T;% Time vector
        
        % Convert the audio signal to a frequency-domain representation using 30 ms
        % windows with 15 ms overlap. Because the input is real and therefore the
        % spectrum is symmetric, you can use just one side of the frequency domain
        % representation without any loss of information. Convert the complex
        % spectrum to the magnitude spectrum: phase information is discarded
        % when calculating mel frequency cepstral coefficients (MFCC).
        [S,~,frames_out(i).t_for_spect] = stft(frames_out(i).Signal_frame,Fs, ...
            "Window",hamming(Param.WindowLenSamp,"periodic"), ...
            "OverlapLength",Param.noverlap, ...
            "FrequencyRange","onesided");
        frames_out(i).PowerSpectrum = S.*conj(S);
        frames_out(i).t_for_spect = frames_out(i).t_for_spect + frames(i).start_time;
        
        [n,~] =size(FramedSig);
        [Frames_with_vocal_phoneme] = ZCR_and_ENG(FramedSig,Param,frames_out(i).segment_speaker);
        if isempty(Frames_with_vocal_phoneme)
            continue
        end
        
        frames_out(i).F1 = [];frames_out(i).F2 = [];frames_out(i).F3 = [];%frames_out(i).F4 = [];
        Voice_thresh = 0.3;
        frames_out(i).LPC_mat = [];
        for j = 1:n
%             try
                [Formants,LPc_dB,frames_out(i).F]=estimatePhonemeFormants(...
                    FramedSig(j,:),Fs,"h",flag);
                frames_out(i).LPC_mat = [frames_out(i).LPC_mat LPc_dB];
                
%             catch ME
%                 fprintf('estimatePhonemeFormants without success: %s\n', ME.message);
%                 Formants = [NaN NaN NaN]';
%                 continue;  % Jump to next iteration of: for i
%             end
            [frames_out(i).pitch(j),frames_out(i).Voice(j)]=sift(FramedSig(j,:),Fs,Voice_thresh,flag);
            
            if any(j == Frames_with_vocal_phoneme) || frames_out(i).Voice(j)>Voice_thresh
                % need to take out the unvoiced segments!!!!!!!!!!!!!!!frames_out(i).Voice(j)>Voice_thresh
                
                frames_out(i).F1 = [frames_out(i).F1 Formants(1)];
                frames_out(i).F2 = [frames_out(i).F2 Formants(2)];
                frames_out(i).F3 = [frames_out(i).F3 Formants(3)];
                %             frames_out(i).F4 = [frames_out(i).F4 Formants(4)];
            else
                frames_out(i).F1 = [frames_out(i).F1 NaN];
                frames_out(i).F2 = [frames_out(i).F2 NaN];
                frames_out(i).F3 = [frames_out(i).F3 NaN];
                %             frames_out(i).F4 = [frames_out(i).F4 NaN];
            end
        end
    end
    if strcmp( frames(i).speakreLabel, "Child") && VTLN==1
        
        warpedFreqs = vtln(freqs, "symmetric", alpha);
        %         freq2time -isreal(warpedFreqs(end))
        P1 = warpedFreqs;
        P1(2:end-1) = warpedFreqs(2:end-1)/2;    % Divide by 2 to correct for amplitude.
        % it is opposite of line 17 in 'dofft'
        P2 = (length(frames_out(i).Signal_frame))*[P1;flipud(conj(P1(2:length(warpedFreqs))))]; % artificially - generate the mirror image of the signal.
        frames_out(i).warpedSignal = real(ifft(P2));                    % get the time domain signal using ifft command.
        %
        %         conjWarpedFreqs = conj(warpedFreqs(end-1:-1:2));
        %         fullFreq = [warpedFreqs; conjWarpedFreqs];
        %          = real(ifft(fullFreq));
        % remove echo
        %         [Rmm,lags] = xcorr(warpedSignal);
        %
        %         Rmm = Rmm(lags>0);
        %         lags = lags(lags>0);
        %         figure
        %         plot(lags/Fs,Rmm)
        %         xlabel('Lag (s)')
        %          [~,dl] = findpeaks(Rmm,lags,"MinPeakHeight",0.3);
        %         % filtfilt
        %         frames_out(i).warpedSignal = filter(1,[1 zeros(1,floor(dl(end))-1) 0.8],warpedSignal);
        
        [S,frames_out(i).FWarped,frames_out(i).t_for_spectWarped] = stft(frames_out(i).warpedSignal,Fs, ...
            "Window",hamming(Param.WindowLenSamp,"periodic"), ...
            "OverlapLength",Param.noverlap, ...
            "FrequencyRange","onesided");
        frames_out(i).PowerSpectrumWarped = S.*conj(S);
        frames_out(i).t_for_spectWarped = frames_out(i).t_for_spectWarped + frames(i).start_time;
        
        FramedSig = enframe(frames_out(i).warpedSignal ,round(Param.noverlap) );
        
        [n,~] =size(FramedSig);
        %         [Frames_with_vocal_phonemeWarped] = ZCR_and_ENG(FramedSig,Param,frames_out(i).segment_speaker);
        %         vocal_frames = find(NRG>(mean(NRG)-(sqrt(var(NRG))/2)) & ZCR<(mean(ZCR)+(sqrt(var(ZCR))/2)));
        
        frames_out(i).F1Warped = [];frames_out(i).F2Warped = [];frames_out(i).F3Warped = [];frames_out(i).LPC_matWarped = [];
        for j = 1:n
            try
                [Formants,LPc_dBWarped,frames_out(i).FWarped_LPC]=estimatePhonemeFormants(...
                    FramedSig(j,:),Fs,"h",flag);
                frames_out(i).LPC_matWarped = [frames_out(i).LPC_matWarped LPc_dBWarped];
            catch ME
                fprintf('estimatePhonemeFormants without success: %s\n', ME.message);
                Formants = [NaN NaN NaN]';
                continue;  % Jump to next iteration of: for i
            end
            %             [Formants,LPc_dBWarped,frames_out(i).FWarped_LPC]=estimatePhonemeFormants(...
            %                 FramedSig(j,:),Fs,"h",flag);
            
            [frames_out(i).pitchWarped(j),frames_out(i).VoiceWarped(j)]=sift(FramedSig(j,:),Fs,Voice_thresh,flag);
            if  any(j == Frames_with_vocal_phoneme)  || frames_out(i).VoiceWarped(j)>Voice_thresh
                % need to take out the unvoiced segments!!!!!!!!!!!!!!!frames_out(i).VoiceWarped(j)>Voice_thresh
                %                 Formants=estimatePhonemeFormants(...
                %                     FramedSig(j,:),Fs,"h",flag);
                frames_out(i).F1Warped = [frames_out(i).F1Warped Formants(1)];
                frames_out(i).F2Warped = [frames_out(i).F2Warped Formants(2)];
                frames_out(i).F3Warped = [frames_out(i).F3Warped Formants(3)];
                %             frames_out(i).F4 = [frames_out(i).F4 Formants(4)];
            else
                frames_out(i).F1Warped = [frames_out(i).F1Warped NaN];
                frames_out(i).F2Warped = [frames_out(i).F2Warped NaN];
                frames_out(i).F3Warped = [frames_out(i).F3Warped NaN];
                %             frames_out(i).F4 = [frames_out(i).F4 NaN];
            end
        end
        frames_out(i).F1Warped  = frames_out(i).F1Warped(1:end-1);
        frames_out(i).F2Warped  = frames_out(i).F2Warped(1:end-1);
        frames_out(i).F3Warped  = frames_out(i).F3Warped(1:end-1);
        
    end %strcmp( frames(i).speakreLabel, "Child") && VTLN==1
end % end the for loop of segments

end

