function [Freq,outAR]= formants(Si, p, AR, Fs, Lags)
% function FORMANTS - formants estimation of given signal or AR coefficients.
% [Freq,outAR] = formants(Si, p, [], Fs, Lags);
% OR [Freq,outAR] = formants([[], 0, AR, Fs, Lags);
% input: Si - time signal.
% p - AR order.
% AR - AR coefficients.
% This function estimates formants in two ways. First when a signal is given
% and the required AR estimated order. The second way is input directly the
% AR coefficients.
% Fs - sampling frequency.
% Lags - 2*Fs/Lags is the frequency accurecy. Lags must be power of 2.
if (max(size(Si))>0),
    if p==0, error('Wrong p value'), end
    AR = aryule(Si, p);
elseif (max(size(AR))==0),
    error('Wrong AR value')
end
[n,m] =size(AR);
if (n>m),
    AR = AR';
else
    n = m;
end
S=[AR zeros(1, Lags-n)];
S1=abs( fft(S));
S1=S1(1:Lags/2);
S2=diff(S1);
Freq= 1+ find( diff( sign(S2))==2);
f=linspace(0,Fs/2,Lags/2);
Freq = f(Freq);
outAR = AR;
end