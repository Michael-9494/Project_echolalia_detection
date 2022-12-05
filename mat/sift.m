function [f,Voice]=sift(signal,Fs)
%----------------------------------------------------------------------
% based on:
% J.D. Markel, "The SIFT algorithm for fudamental frequency estimation"
% IEEE Tran. Audio Electrooacoust., Vol. AU-20, pp. 367-377, Dec. 1972.
%----------------------------------------------------------------------
signal=signal-mean(signal); % Clean signal from DC level
% Prefiltering: LPF whith A cutoff at 0.1KHz.
[B,A]=cheby1(3,2,800/Fs);
Si=filter(B,A,signal);
% Step=floor(Fs/2e3); % Down sampling to 2Khz - for 10KHz ; NOT USED HERE
Step=1;
nFs=Fs/Step; %new sampling rate, since there is no down sampling it stays the same
Len=max(size(Si));
NewLen=floor(Len/Step);
Si1=Si(1+(0:NewLen-1)*Step);
% Determination of the inverse filter
N=NewLen-4;
p0=Si1(1:N)*Si1(1:N)';
p1=Si1(1:N)*Si1(2:N+1)';
p2=Si1(1:N)*Si1(3:N+2)';
p3=Si1(1:N)*Si1(4:N+3)';
p4=Si1(1:N)*Si1(5:N+4)';
% Solving the autocorrelation equations
R = [p0 p1 p2 p3; p1 p0 p1 p2; p2 p1 p0 p1; p3 p2 p1 p0];
P = -[p1 p2 p3 p4]';
a = inv(R) * P;
Si2=filter([1 -a'],1,Si1); % Inverse filter
Xr=xcorr(Si2);
Xr1=Xr(NewLen:1.5*NewLen);
Lxr=max(size(Xr1));
Xr1=Xr1/Xr1(1); %normlizing the correlation
% Set to zero all values until the first negativ value
Xr1=Xr1.*(Xr1>0);
Zcr=min(find(Xr1==0));
Xr1(1:Zcr)=zeros(1,Zcr);
M=find(Xr1==max(Xr1)); %looks for the first peak
Voice_prob=Xr1(M);

if max(size(M))>1 || Voice_prob<=0.4
    f=0; Voice=0;
else
    f=nFs/M; Voice=Xr1(M);
end
end