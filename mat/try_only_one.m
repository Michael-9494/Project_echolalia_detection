clear;close all;clc
flag = 0;
[Signal,Fs]=audioread("me.wav");
r = 3;
Signal1 = decimate(Signal,r);
Fs = Fs/r;
Param = struct();
% start with feature extraction
Param.alpha=0.98; % for pre emphasis
Param.WindowLength=20*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
% Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);
soundsc(Signal1,Fs);

% PreProcess make all the pre processing wotk we need to do
% the steps are based on DC removal, HP filtering.
[ProcessedSig,FramedSig] = PreProcess(...
    Signal1,Fs,Param.alpha,Param.WindowLength,Param.Overlap);
T = 1/Fs;             % Sampling period
Len = length(ProcessedSig);
t =  (0:Len-1)*T;% Time vector]

P2 = fft(ProcessedSig)/length(ProcessedSig);% perform fft > it gives double-spectrum and distribute energy
freqs = P2(1:ceil(length(ProcessedSig)/2)+1);              % get one-side spectrum
freqs(2:end-1) = 2*freqs(2:end-1);    % Multiple by 2 as a correction for amplitude
f=[0:1:ceil(Len/2)-1]*Fs/round(Len); % Map the frequency bin to frequency (Hz)


%         [~,FramedSig] = PreProcess(...
%             frames_out(i).Signal_frame,Fs,Param.alpha,Param.WindowLength,Param.Overlap);


[S,F_for_spect,t_for_spect] = stft(ProcessedSig,Fs, ...
    "Window",hamming(Param.WindowLenSamp,"periodic"), ...
    "OverlapLength",Param.noverlap, ...
    "FrequencyRange","onesided");
PowerSpectrum = S.*conj(S);



[n,~] =size(FramedSig);
F1 = [];F2 = [];F3 = [];%frames_out(i).F4 = [];

LPC_mat = [];
for j = 1:n
    [pitch(j),Voice(j)]=sift(FramedSig(j,:),Fs);
    [Formants,LPc_dB,F_LPC]=estimatePhonemeFormants(...
        FramedSig(j,:),Fs,"h",flag);
    LPC_mat = [LPC_mat LPc_dB];

    if Voice(j)>0.4
        %         need to take out the unvoiced segments!!!!!!!!!!!!!!!

        F1 = [F1 Formants(1)];
        F2 = [F2 Formants(2)];
        F3 = [F3 Formants(3)];

    else
        F1 = [F1 NaN];
        F2 = [F2 NaN];
        F3 = [F3 NaN];
        %             frames_out(i).F4 = [frames_out(i).F4 NaN];
    end
end




fh = figure(2);
fh.WindowState = 'maximized';

figure(2);subplot(4,1,1)
plot(t,ProcessedSig)
title('speech signal ' );
xlabel('time[s]');ylabel('amp'); axis tight

figure(2);subplot(4,1,2)
plot(f,abs(freqs(1:end-1)));grid on
title('speech spectrum ' );
xlabel('F[Hz]');ylabel('amp'); axis tight


figure(2);subplot(4,1,3)
surf(t_for_spect,F_for_spect,...
    20*log10(PowerSpectrum),"DisplayName","Power_{Spectrum}","EdgeColor","none");
view([0,90]);hold on;
axis([t_for_spect(1) t_for_spect(end) F_for_spect(1) F_for_spect(end)])
plot(t_for_spect,F1,"b","DisplayName","F_1");
hold on;
plot(t_for_spect,F2,"r","DisplayName","F_2");
hold on;
plot(t_for_spect,F3,"k","DisplayName","F_3");
hold on;
plot(t_for_spect,pitch,"g","DisplayName","F_0");
xlabel('Time (s)');hold on
hold on;ylabel('Frequency (Hz)');title("Child LPC")
legend();

figure(2);subplot(4,1,4)
surf(t_for_spect,F_LPC,(LPC_mat),"DisplayName","LPC_{Spectrum}","EdgeColor","none");
view([0,90]); hold on
axis([t_for_spect(1) t_for_spect(end) F_LPC(1) F_LPC(end)])
xlabel('Time (s)');hold on
plot(t_for_spect,F1,"b","DisplayName","F_1");
hold on;
plot(t_for_spect,F2,"r","DisplayName","F_2");
hold on;
plot(t_for_spect,F3,"k","DisplayName","F_3");
hold on;
plot(t_for_spect,pitch,"g","DisplayName","F_0");
hold on;ylabel('Frequency (Hz)');title("Therapist LPC")
legend();