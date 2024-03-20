function [B,M]=ltp(sp)
% Given a vector c of audio samples, we can add in a pitch component identified by an amplitude
% β and lag M as shown in Equation (6.33) to generate a spiky signal x which includes
% pitch:
% x(n) = c(n) + βx(n − M). (6.33)
% β scales the amplitude of the pitch component and the lag M corresponds to the primary
% pitch period.
n=length(sp);
%upper & lower pitch limits (Fs~8kHz-16kHz)
pmin=50; pmax=200;
sp2=sp.^2; %pre-calculate
for M=pmin:pmax
    e_del=sp(1:n-M);
    e=sp(M+1:n);
    e2=sp2(M+1:n);
    E(1+M-pmin)=sum((e_del.*e).^2)/sum(e2);
end
%---find M, the optimum pitch period
[null, M]=max(E);
M=M+pmin;
%---find B, the pitch gain
e_del=sp(1:n-M);
e=sp(M+1:n);
e2=sp2(M+1:n);
B=sum(e_del.*e)/sum(e2);
end