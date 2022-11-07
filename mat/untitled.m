clear all
clc
close all


[Signal,Fs]=audioread('sample2.wav');
Signal = Signal(:,2);
Signal = Signal(1:end);

T = 1/Fs;             % Sampling period
L = length(Signal);             % Length of signal
t = (0:L-1)*T;        % Time vector
% Signal = Signal(t'<8 );
L = length(Signal);             % Length of signal

figure
plot(t(1:length(Signal)),Signal)
title('speech signal X(t)');xlabel('time[s]');ylabel('amp')
grid on; axis tight

% soundsc(Signal,Fs);
% Compute the Fourier transform of the signal.
Y = fft(Signal);
% Compute the two-sided spectrum P2. Then compute the single-sided spectrum
% P1 based on P2 and the even-valued signal length L.
P2 = abs(Y/L);P1 = P2(1:L/2+1);P1(2:end-1) = 2*P1(2:end-1);
% Define the frequency domain f and plot the single-sided amplitude spectrum P1.
% The amplitudes are not exactly at 0.7 and 1, as expected, because of the
% added noise. On average, longer signals produce better frequency approximations.
f = Fs*(0:(L/2))/L;
figure
plot(f,P1)
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% soundsc(Signal,Fs);
alpha=0.98;
WindowLength=30*10^-3;  % 40 [mS] window with 30 ms overlap
Overlap=75;             %  %frame rate of 10 ms
[ProcessedSig,FramedSig] = PreProcess(Signal,Fs,alpha,WindowLength,Overlap);
% soundsc(ProcessedSig,Fs);
figure
plot(t(1:L),ProcessedSig)
title('PreProcessed speech signal X(t)');xlabel('time[s]');ylabel('amp')
grid on; axis tight

Y = fft(ProcessedSig);P2 = abs(Y/L);P1 = P2(1:L/2+1);P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
figure
plot(f,P1)
title('PreProcessed Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%%
Nx = length(FramedSig(1,:));
nsc = floor(Nx/4.5);
nov = floor(nsc/2);
nff = max(256,2^nextpow2(nsc));


[pxx1,f] = pwelch(FramedSig(1,:),hamming(nsc),nov,nff,Fs);
figure
plot(f,10*log10(pxx1));grid on
xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')
%%
% do stft with 50 % overlap
figure
window=hamming(Nx);
noverlap = ((Overlap)*(WindowLength*Fs))/100;
spectrogram(ProcessedSig,window,noverlap,nff,Fs,'yaxis')
title(['Spectrogram. window ' num2str(WindowLength*10^3) '[mS]->N=' num2str(Nx) '. 50% overlap'])
colormap jet

%%
[Idx,idx_vec_start,idx_vec_end]=FindWordIdx(FramedSig,Fs,WindowLength,Overlap);

figure(200),subplot(3,1,1)
plot(t(1:L),ProcessedSig);xlim([0 t(L)]);
ylim([min(ProcessedSig) max(ProcessedSig)]);
hold on
% line([t(Idx(1)) t(Idx(1))],ylim,'color','m','linewidth',1.2)
% line([t(Idx(2)) t(Idx(2))],ylim,'color','m','linewidth',1.2)
title 'signal - marker the speech '; xlabel 'Time [S]' ; ylabel 'Amplitude'
hold on
for i=1:length(idx_vec_start)
    line([t(idx_vec_start(i)) t(idx_vec_start(i))],ylim,'color','r','linewidth',1.2)
    hold on
    line([t(idx_vec_end(i)) t(idx_vec_end(i))],ylim,'color','b','linewidth',1.2)
end
legend('speech','start ','end','start ','end','start ','end');grid on

dt = 40*10^-3; % minimum time above threshold 'eta' [mS]
eta =400;
winlen = 50*10^-3; % 30 [mS]
% t=[0:length(ProcessedSig)-1]*1/Fs;
figure(200),subplot(3,1,2), plot(t,ProcessedSig); hold on;grid on

for j=1:length(idx_vec_start)
    [seg_ind,delta]=segmentation1(ProcessedSig,winlen,eta,dt,Fs,[idx_vec_start(j) idx_vec_end(j)]);

    %  ProcessedSig = ProcessedSig(Idx(1):Idx(2));
    
    delta=delta(1:length(ProcessedSig));
    % show results:
    figure(200),subplot(3,1,2),
    % show segmentation lines:
    for i=1:length(seg_ind)
        line([t(seg_ind(i)) t(seg_ind(i))],ylim,'color','m','linewidth',1.2)
    end
    xlim([0 t(end)])
    title 'Segmentation - Shalom speech signal' ;xlabel 'Time [S]' ;ylabel 'Amplitude'
    legend('speech','segments');  axis tight;
    hold on

    figure(200),subplot(3,1,3),plot(t,delta); hold on
    yline(eta,'r','linewidth',1.2)
    for i=1:length(seg_ind)

        line([t(seg_ind(i)) t(seg_ind(i))],ylim,'color','m','linewidth',1.2)
    end
    xlim([0 t(end)]);grid on
    title 'Spectral error measure' ;xlabel 'Time [S]' ;ylabel 'Amplitude'
    legend('\Delta_1',['\eta=' num2str(eta)],'segments');
end