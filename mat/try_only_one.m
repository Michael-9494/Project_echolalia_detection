clear;close all;clc
flag_formants = 0;
flag_sift = 0 ;
flag_sound = 1;
[Signal,Fs]=audioread("igulim.wav");
% Signal = Signal(round(0.4*Fs):round(1.2*Fs)-1);
% soundsc(Signal_th,Fs);
% Signal_ch = Signal(round(1.7*Fs):round(2.4*Fs)-1);
% soundsc(Signal_ch,Fs);


% r = 2;
% Signal = decimate(signal,r);
% Fs = Fs/r; % new sampling rate
p = Fs/1000+2;
Param = struct();
% start with feature extraction
Param.alpha= 15/16; % for pre emphasis0.96
Param.WindowLength=20*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
% Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);
if flag_sound
    soundsc(Signal,Fs);
end
% PreProcess make all the pre processing wotk we need to do
% the steps are based on DC removal, HP filtering.
% FramedSig = enframe(ProcessedSig,hamming(Param.WindowLenSamp,"periodic") ,round(Param.noverlap) );
[ProcessedSig,FramedSig] = PreProcess(...
    Signal,Fs,Param.alpha,Param.WindowLength,Param.Overlap);
if flag_sound
    pause(3)
    soundsc(ProcessedSig,Fs);
end
T = 1/Fs;             % Sampling period
Len = length(ProcessedSig);
t = (0:Len-1)*T;% Time vector] 0.22 +
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
% t_for_spect = 0.22 + t_for_spect ;


warpedFreqs = vtln(freqs, "symmetric", 1.2);
%         freq2time -isreal(warpedFreqs(end))
P1 = warpedFreqs;
P1(2:end-1) = warpedFreqs(2:end-1)/2;    % Divide by 2 to correct for amplitude.
P2 = (length(ProcessedSig))*[P1;flipud(conj(P1(2:length(warpedFreqs)-1)))]; % artificially - generate the mirror image of the signal.
warpedSignal = real(ifft(P2));
if flag_sound
    pause(5)
    
    soundsc(warpedSignal,Fs);
end
[SWarped,F_for_spectWarped,t_for_spectWarped] = stft(warpedSignal,Fs, ...
    "Window",hamming(Param.WindowLenSamp,"periodic"), ...
    "OverlapLength",Param.noverlap, ...
    "FrequencyRange","onesided");
PowerSpectrumWarped = SWarped.*conj(SWarped);
% t_for_spectWarped = 0.22 +t_for_spectWarped ;

[vocal_frames,NRG,ZCR1] = ZCR_and_ENG(FramedSig);
ZCRr = ZCR(FramedSig);
figure
subplot(2,1,1)
plot(ZCR1,"DisplayName","ZCR"),xlabel('frame number');
ylabel('Zero-crossing rate'),title("Signal ZCR");yline(mean(ZCR1)+(sqrt(var(ZCR1))/2),'--m',"DisplayName","ZCR mean");
subplot(2,1,2),plot(NRG,"--","DisplayName","NRG");
title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
grid on; hold on
yline(mean(NRG)-(sqrt(var(NRG))/2),'--m',"DisplayName","NRG mean");%yline(th1,'b');yline(th2,'g')
legend();
hold on

[n,~] =size(FramedSig);
% f0 = pitch(ProcessedSig,Fs,'WindowLength',Param.WindowLenSamp,...
%     'OverlapLength',Param.noverlap,"Range",[90 600]);
% f0Warped = pitch(warpedSignal,Fs,'WindowLength',Param.WindowLenSamp,...
%     'OverlapLength',Param.noverlap,"Range",[90 600]);
F1 = [];F2 = [];F3 = [];%frames_out(i).F4 = [];
Voice_thresh = 0.2;
LPC_mat = [];
F1Warped = [];F2Warped = [];F3Warped = [];%frames_out(i).F4 = [];
LPC_matWarped = [];
FramedSigWarped = enframe(warpedSignal ,round(Param.noverlap) );

for j = 1:n
    [Formants,LPc_dB,F_LPC]=estimatePhonemeFormants(...
        FramedSig(j,:),Fs,"h",flag_formants);
    LPC_mat = [LPC_mat LPc_dB];
    [pitchh(j),Voice(j)]=sift(FramedSig(j,:),Fs,Voice_thresh,flag_sift);
    
    if any(j == vocal_frames) || Voice(j)>Voice_thresh
        % need to take out the unvoiced segments!!!!!!!!!!!!!!!  
        
        F1 = [F1 Formants(1)];
        F2 = [F2 Formants(2)];
        F3 = [F3 Formants(3)];
        %             frames_out(i).F4 = [frames_out(i).F4 Formants(4)];
    else
        F1 = [F1 NaN];
        F2 = [F2 NaN];
        F3 = [F3 NaN];
        %                 %             frames_out(i).F4 = [frames_out(i).F4 NaN];
    end
    
    [FormantsWarped,LPc_dBWarped,F_LPCWarped]=estimatePhonemeFormants(...
        FramedSigWarped(j,:),Fs,"h",flag_formants);
    LPC_matWarped = [LPC_matWarped LPc_dBWarped];
    [pitchhWarped(j),VoiceWarped(j)]=sift(FramedSigWarped(j,:),Fs,Voice_thresh,flag_sift);
    
    if any(j == vocal_frames) || Voice(j)>Voice_thresh
        % need to take out the unvoiced segments!!!!!!!!!!!!!!! VoiceWarped(j)>Voice_thresh 
        
        F1Warped = [F1Warped FormantsWarped(1)];
        F2Warped = [F2Warped FormantsWarped(2)];
        F3Warped = [F3Warped FormantsWarped(3)];
        %             frames_out(i).F4 = [frames_out(i).F4 Formants(4)];
    else
        F1Warped = [F1Warped NaN];
        F2Warped = [F2Warped NaN];
        F3Warped = [F3Warped NaN];
        %                 %             frames_out(i).F4 = [frames_out(i).F4 NaN];
    end
end

% get the time domain signal using ifft command.
%


fh = figure(2);
fh.WindowState = 'maximized';

figure(2);subplot(4,2,1)
plot(t,ProcessedSig)
title('speech signal ' );
xlabel('time[s]');ylabel('amp'); axis tight

figure(2);subplot(4,2,3)
plot(f,abs(freqs(1:end-1)));grid on
title('speech spectrum ' );
xlabel('F[Hz]');ylabel('amp'); axis tight


figure(2);subplot(4,2,5)
surf(t_for_spect,F_for_spect,...
    20*log10(abs(S)),"DisplayName","Power_{Spectrum}","EdgeColor","none");
view([0,90]);hold on;
axis([t_for_spect(1) t_for_spect(end) F_for_spect(1) F_for_spect(end)])
plot(t_for_spect,F1,"b","DisplayName","F_1");
hold on;
plot(t_for_spect,F2,"r","DisplayName","F_2");
hold on;
plot(t_for_spect,F3,"k","DisplayName","F_3");
hold on;
plot(t_for_spect,pitchh,"m","DisplayName","F_0");
% hold on;
% plot(t_for_spect,f0,"m","DisplayName","F_0");
xlabel('Time (s)');hold on;ylim([0 8000]);
hold on;ylabel('Frequency (Hz)');title("Child LPC")
% legend();

figure(2);subplot(4,2,7)
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
plot(t_for_spect,pitchh,"m","DisplayName","F_0");
% hold on;
% plot(t_for_spect,f0,"m","DisplayName","F_0");
hold on;ylabel('Frequency (Hz)');title("Therapist LPC");ylim([0 8000]);
% legend();





figure(2);subplot(4,2,2)
plot(t,warpedSignal)
title('warped speech signal ' );
xlabel('time[s]');ylabel('amp'); axis tight

figure(2);subplot(4,2,4)
plot(f,abs(P1(1:end-1)));grid on
title('speech spectrum ' );
xlabel('F[Hz]');ylabel('amp'); axis tight


figure(2);subplot(4,2,6)
surf(t_for_spectWarped,F_for_spectWarped,...
    20*log10(abs(SWarped)),"DisplayName","Power_{Spectrum}","EdgeColor","none");
view([0,90]);hold on;
axis([t_for_spectWarped(1) t_for_spectWarped(end) F_for_spectWarped(1) F_for_spectWarped(end)])
plot(t_for_spectWarped,F1Warped,"b","DisplayName","F_1");
hold on;
plot(t_for_spectWarped,F2Warped,"r","DisplayName","F_2");
hold on;
plot(t_for_spectWarped,F3Warped,"k","DisplayName","F_3");
hold on;
plot(t_for_spectWarped,pitchhWarped,"m","DisplayName","F_0");
% hold on;
% plot(t_for_spectWarped,f0Warped,"m","DisplayName","F_0");
xlabel('Time (s)');hold on
hold on;ylabel('Frequency (Hz)');title("Child LPC");ylim([0 8000]);
% legend();

figure(2);subplot(4,2,8)
surf(t_for_spectWarped,F_LPCWarped,(LPC_matWarped),"DisplayName","LPC_{Spectrum}","EdgeColor","none");
view([0,90]); hold on
axis([t_for_spectWarped(1) t_for_spectWarped(end) F_LPCWarped(1) F_LPCWarped(end)])
xlabel('Time (s)');hold on
plot(t_for_spectWarped,F1Warped,"b","DisplayName","F_1");
hold on;
plot(t_for_spectWarped,F2Warped,"r","DisplayName","F_2");
hold on;
plot(t_for_spectWarped,F3Warped,"k","DisplayName","F_3");
hold on;
plot(t_for_spectWarped,pitchhWarped,"m","DisplayName","F_0");
% hold on;
% plot(t_for_spectWarped,f0Warped,"m","DisplayName","F_0");
hold on;ylabel('Frequency (Hz)');title("Warped Child LPC");ylim([0 8000]);
% legend();
% soundsc(warpedSignal,Fs);

figure(10)
plot(Voice,"b","DisplayName","Voiced_{P%}");hold on
plot(VoiceWarped,"r","DisplayName","Warped Voiced_{P%}")
legend();ylim([0 1])

% [temp]=dynamictimewarping(Signal_ch,Signal_th)
a1 = Signal_ch;a2 = Signal;
[d,i1,i2] = dtw(a1,a2);

a1w = a1(i1);
% soundsc(a1w,Fs)
a2w = a2(i2);

t = (0:numel(i1)-1)/Fs;
% duration = t(end)
figure
plot(t,a1w,"DisplayName","Signal_{ch}")
title(' Warped');hold on
plot(t,a2w,"DisplayName","Signal_{th}");legend();
xlabel('Time (seconds)')