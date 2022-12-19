clear all
clc
% 
% 
% [Signal,Fs]=audioread('sample.wav');
% Signal = Signal(:,2);
% 
% % Fs = 1000;            % Sampling frequency
% T = 1/Fs;             % Sampling period
% L = length(Signal);             % Length of signal
% t = (0:L-1)*T;        % Time vector
% 
% figure
% plot(t,Signal)
% title('speech signal X(t)');xlabel('time[s]');ylabel('amp')
% grid on; axis tight
% 
% % soundsc(Signal,Fs);
% % Compute the Fourier transform of the signal.
% Y = fft(Signal);
% % Compute the two-sided spectrum P2. Then compute the single-sided spectrum
% % P1 based on P2 and the even-valued signal length L.
% P2 = abs(Y/L);P1 = P2(1:L/2+1);P1(2:end-1) = 2*P1(2:end-1);
% % Define the frequency domain f and plot the single-sided amplitude spectrum P1.
% % The amplitudes are not exactly at 0.7 and 1, as expected, because of the
% % added noise. On average, longer signals produce better frequency approximations.
% f = Fs*(0:(L/2))/L;
% figure
% plot(f,P1)
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% 
% % soundsc(s,Fs);
% alpha=0.98;
% WindowLength=30*10^-3;  % 30 [mS] window
% Overlap=50;             % 50% overlap
% [ProcessedSig,FramedSig] = PreProcess(Signal,Fs,alpha,WindowLength,Overlap);
% % h=[1, -0.9375];
% % y=filter(h, 1, s);
% % soundsc(y,Fs);
% 
% 
% 
% %%
% y=chirp([0:0.001:5],0,5,500); %construct a frequency chirp
% z=[y,y(length(y):-1:1),y]; %make a zig-zag
% % First let’s take one huge FFT and plot this as shown below:
% figure
% f=abs(fft(z, 8192));
% plot(f(1:4096));
% s=spectrogram(z, 1024);
% waterfall(abs(s)');
% 
% 
% %%
% 
% ps=log(abs(fft(Signal.*hamming(length(Signal)))));
% figure
% 
% plot(abs(ifft( ps )));
% cx=cceps(Signal.*hamming(length(Signal)));
% stem(abs(cx));
% axis tight;
% xlabel('Cepstral coefficient')
% ylabel('Log magnitude')
% 
% 
% %% Sound generation
% clear all
% clc
% 
% note=tonegen(440, 16000, 2);
% % soundsc(note, 16000);
% 
% % 440*(2^(1/12))^12=440*2=880
% 
% Fs=8000; %define sample frequency
% Tt=[0:1/Fs:4]; %create array of sample times
% ModF=1000+200*sin(Tt*2*pi);
% 
% gensnd=[]; %initialise an empty array
% for t=0:0.1:4
%     fr=ModF(1+floor(t*Fs));
%     % use floor() since index must be integer
%     gensnd=[gensnd,tonegen(fr,8000,0.1)];
%     % concatenates new tones to end of gensnd
% end
% % soundsc(gensnd,Fs); % listen then view it
% figure
% spectrogram(gensnd,128,0,128,Fs,'yaxis');
% 
% figure
% gensnd2=freqgen(ModF, Fs);
% % soundsc(gensnd2,Fs); % listen, then view it
% spectrogram(gensnd2,128,0,128,Fs,'yaxis');
% 
% 
% freq=[440*(1+zeros(1,1000)), 415.2*(1+zeros(1,1000)),392*(1+zeros(1,1000))];
% music=freqgen(freq, 8000);
% % soundsc(music, 8000);



% As an example, we could create a 256-element array of frequencies spaced equally on
% the mel scale from 0 to 4 kHz and then compute the spreading function for 
mmax=f2mel(4000);
melarray=[0:mmax/255:mmax]; %256 elements
hzarray=mel2f(melarray);
% a frequency at 1.2 kHz as follows:
% [idx idx]=min(abs(hzarray-800));
figure
for idx =1:256
    spread=spread_mel(hzarray,idx,10,4000);
    
    plot(f2mel([1:10]*4000/10),spread,'o-')

    hold on
end
%%

clear all
clc
[Signal,Fs]=audioread('sample.wav');
Signal = Signal(:,2);
Signal = Signal(round(1.4*Fs):round(2*Fs));
% numHops = floor((r-winLength)/hopLength) + 1
alpha=0.98;
WindowLength=20*10^-3;  % 30 [mS] window
Overlap=50;             % 50% overlap
[ProcessedSig,FramedSig] = PreProcess(Signal,Fs,alpha,WindowLength,Overlap);

Ws=25*10^(-3)*Fs;
Ol=((Overlap)*Ws)/100;
L=floor((length(ProcessedSig)-Ws)/Ol)+1
N=13;
tic
ccs=zeros(N,L);
for n=1:L
    seg=ProcessedSig(1+(n-1)*Ol:Ws+(n-1)*Ol);
    ccs(:,n)=mfcc_model(seg.*hamming(Ws),20,N,Fs);
end
ccs = ccs';
timee = toc;
waterfall([1:L]*length(Signal)/(L*Fs),[1:N],ccs')
xlabel('Time, s')
ylabel('Amplitude')
ylabel('Band')
zlabel('Amplitude')


% compare to matlab function

S = stft(Signal,"Window",hamming(Ws),"OverlapLength",512,"Centered",false);
[coeffs,delta,deltaDelta,loc] = mfcc(S,Fs,"LogEnergy","Ignore");
figure
waterfall([1:length(coeffs(:,1))]*length(Signal)/(length(coeffs(:,1))*Fs),[1:N],coeffs')
xlabel('Time, s')
ylabel('Amplitude')
ylabel('Band')
zlabel('Amplitude')
%% This is for testing the MFCC model
clear all
clc


% [Signal,Fs]=audioread('sample.wav');r
[s,fs]=audioread('me_and_nicole.WAV'); %load speech
s = s(:,1);
s = s(round(1.4*fs):round(4*fs));
Ws=1024;
Ol=512;
L=floor((length(s)-Ol)/Ol);
N=12;

ccs=zeros(N,L);
for n=1:L
    seg=s(1+(n-1)*Ol:Ws+(n-1)*Ol);
    ccs(:,n)=mfcc_model(seg.*hamming(1,Ws),20,N,fs);
end
%  Masking effect of speech 137
figure
xlabel('Time, s')
ylabel('Amplitude')
ylabel('Band')
zlabel('Amplitude')
waterfall([1:L]*length(s)/(L*fs),[1:N],ccs)