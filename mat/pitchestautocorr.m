function [pitchfrequency]=pitchestautocorr(S,FS)

% SEGLEN=FS*20*10^(-3);

% S1=S(fix(length(S)/2):SEGLEN-1+fix(length(S)/2))';
T=[];
f_max = 400;f_min = 200;
HIGH=fix(FS/f_min);
LOW=fix(FS/f_max);
for k=LOW:1:HIGH
    T=[T sum([zeros(1,k) S ].*[S zeros(1,k) ])];
end

[P,Q]=max(T);
pitchperiod=Q+LOW;
pitchfrequency=FS/pitchperiod;