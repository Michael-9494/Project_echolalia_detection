clear all
clc


[Signal,Fs]=audioread('sample.wav');
Signal = Signal(:,2);

T = 1/Fs;             % Sampling period
L = length(Signal);             % Length of signal
t = (0:L-1)*T;        % Time vector

figure
plot(t,Signal)
title('speech signal X(t)');xlabel('time[s]');ylabel('amp')
grid on; axis tight

% soundsc(Signal,Fs);

x = Signal;
X = x/norm(x);
figure
plot(t,X)
title('speech signal X(t)');xlabel('time[s]');ylabel('amp')
grid on; axis tight

y = rand(size(X));

% Obtain the Welch PSD estimate using the default Hamming window and DFT length.
% The default segment length is 71 samples and the DFT length is the 256 points yielding
% a frequency resolution of 2π/256 rad/sample. Because the signal is real-valued,
% the periodogram is one-sided and there are 256/2+1 points. Plot the Welch PSD estimate.

pxx = pwelch(X);

% pwelch(X)

% Repeat the computation.
% 
% Divide the signal into sections of length nsc=(N_x/4.5). 
%   This action is equivalent to dividing the signal into the longest possible 
%   segments to obtain as close to but not exceed 8 segments with 50% overlap.
% Window the sections using a Hamming window.
% Specify 50% overlap between contiguous sections
% 
% % To compute the FFT, use max(256,2^p) points, where p=(log2nsc).
% 
% Verify that the two approaches give identical results.

Nx = length(X);
nsc = floor(Nx/4.5);
nov = floor(nsc/2);
nff = max(256,2^nextpow2(nsc));


[pxx1,f] = pwelch(x,hamming(nsc),nov,nff,Fs);
figure
plot(f,10*log10(pxx1));grid on
xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')
maxerr = max(abs(abs(pxx1(:))-abs(pxx(:))))

Nx = length(y);
nsc = floor(Nx/4.5);
nov = floor(nsc/2);
nff = max(256,2^nextpow2(nsc));


[pxx1,f] = pwelch(y,hamming(nsc),nov,nff,Fs);
figure
plot(f,10*log10(pxx1));grid on
xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')

% Divide the signal into 8 sections of equal length, with 50% overlap between sections.
% Specify the same FFT length as in the preceding step. Compute the Welch PSD estimate
% and verify that it gives the same result as the previous two procedures.
% ns = 8;
% ov = 0.5;
% lsc = floor(Nx/(ns-(ns-1)*ov));
% figure
% [pxx,f] = pwelch(x,lsc,floor(ov*lsc),nff,Fs);
% figure
% plot(f,10*log10(pxx))
% xlabel('Frequency (Hz)')
% ylabel('PSD (dB/Hz)')
% % maxerr2 = max(abs(abs(pxx1(:))-abs(pxx(:))))



%%

