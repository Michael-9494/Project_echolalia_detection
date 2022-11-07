function cc=mfcc_model(seg, N, M, Fs)
% Do FFT of audio frame seg, map to M MFCCs
% from 0 Hz to Fs/2 Hz, using N filterbanks
% typical values N=26,M=12,Fs=8000,seg~20ms
m_low=0; %mel span lower limit
m_top=f2mel(Fs/2); %mel span upper limit
mdiv=(m_top-m_low)/(N-1); %mel resolution
%Define an array of centre frequencies
xm=m_low:mdiv:m_top;
%Convert this to Hz frequencies
xf=mel2f(xm);
%Quantise to the FFT resolution
xq = floor((length(seg)/2 + 1)*xf/(Fs/2));
%Take the FFT of the speech...
S=fft(seg);
S=abs(2*(S.*S)/length(S));
S=S(1:length(S)/2);
F=[1:length(S)]*(Fs/2)/length(S);

%Compute the mel filterbanks
x1=zeros(1,N);
for xi=1:N
    band=spread_mel(xf,xi,length(S),Fs/2);
    % apply my new VTLN 
    x1(xi)=sum(band.*S');
end
x=log(x1);
%Convert to MFCC using loop (could use matrix)
cc=zeros(1,M);
for xc=1:M
    cc(xc)=sqrt(2/N)*sum(x.*cos(pi*xc*([1:N]-0.5)/N));
end
end