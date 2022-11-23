function [frames_out] = Process_frame(frames,Fs,Param,flag)
%PROCESS_FRAME Summary of this function goes here
%   Detailed explanation goes here


%load the struct containing the speech frames
segments = size(frames);
frames_out = struct();
for i = 1:segments(2)
    % load each speech segment with its labels
    Signal_frame = frames(i).data;
    segment_speaker = frames(i).speakreLabel;
    segment_event = frames(i).event;

    [frames_out(i).ProcessedSig,frames_out(i).FramedSig] = PreProcess(...
        Signal_frame,Fs,Param.alpha,Param.WindowLength,Param.Overlap);
    T = 1/Fs;             % Sampling period
    Len = length(frames_out(i).ProcessedSig);

    frames_out(i).t = frames(i).start_time + (0:Len-1)*T;% Time vector

    %     figure, plot(frames_out(i).t,frames_out(i).ProcessedSig);
    %     title('speech signal ');xlabel('time[s]');ylabel('amp')
    %     grid on; axis tight
    %     hold on

    % Convert the audio signal to a frequency-domain representation using 30 ms
    % windows with 15 ms overlap. Because the input is real and therefore the
    % spectrum is symmetric, you can use just one side of the frequency domain
    % representation without any loss of information. Convert the complex
    % spectrum to the magnitude spectrum: phase information is discarded
    % when calculating mel frequency cepstral coefficients (MFCC).


    [frames_out(i).S,frames_out(i).F,frames_out(i).t_for_spect] = stft(frames_out(i).ProcessedSig,Fs, ...
        "Window",hamming(Param.WindowLenSamp,"periodic"), ...
        "OverlapLength",Param.noverlap, ...
        "FrequencyRange","onesided");
    frames_out(i).PowerSpectrum = frames_out(i).S.*conj(frames_out(i).S);

    %     figure
    %     surf(frames_out(i).t_for_spect,frames_out(i).F,10*log10(frames_out(i).PowerSpectrum),"EdgeColor","none");
    %     view([0,90])
    %     axis([frames_out(i).t_for_spect(1) frames_out(i).t_for_spect(end) frames_out(i).F(1) frames_out(i).F(end)])
    %     xlabel('Time (s)')
    %     ylabel('Frequency (Hz)')
    %     c = colorbar;
    %     c.Label.String = 'Power (dB)';
    %     title("Power Spectrum")

    [n,~] =size(frames_out(i).FramedSig);

    for j = 1:n
        frames_out(i).Formants(j).data=estimatePhonemeFormants(...
            frames_out(i).FramedSig(j,:),Fs,"h",flag);
    end
end
% add the resulting struct
end

