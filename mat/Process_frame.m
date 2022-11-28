function [frames_out] = Process_frame(frames,Fs,Param,flag)
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

        frames_out(i).Signal_frame = frames(i).data;
        frames_out(i).segment_speaker = frames(i).speakreLabel;
        frames_out(i).segment_event = frames(i).event;

        [~,frames_out(i).FramedSig] = PreProcess(...
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
        [S,frames_out(i).F,frames_out(i).t_for_spect] = stft(frames_out(i).Signal_frame,Fs, ...
            "Window",hamming(Param.WindowLenSamp,"periodic"), ...
            "OverlapLength",Param.noverlap, ...
            "FrequencyRange","onesided");
        frames_out(i).PowerSpectrum = S.*conj(S);
        frames_out(i).t_for_spect = frames_out(i).t_for_spect + frames(i).start_time;

        [n,~] =size(frames_out(i).FramedSig);
        [Frames_with_vocal_phoneme,frames_out(i).Signal_Energy,...
            frames_out(i).ZeroCrossingSignal] = ZCR_and_ENG(...
            frames_out(i).FramedSig,Fs,Param.WindowLength,Param.Overlap);
        frames_out(i).F1 = [];frames_out(i).F2 = [];frames_out(i).F3 = [];%frames_out(i).F4 = [];
        for j = 1:n
            if any(Frames_with_vocal_phoneme== j)
                % need to take out the unvoiced segments!!!!!!!!!!!!!!!
                Formants=estimatePhonemeFormants(...
                    frames_out(i).FramedSig(j,:),Fs,"h",flag);
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
end
% add the resulting struct
end

