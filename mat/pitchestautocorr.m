function [pitchfrequency]=pitchestautocorr(S,FS)

SEGLEN=FS*30*10^(-3);

S1=S(fix(length(S)/2):SEGLEN-1+fix(length(S)/2))';
T=[];
HIGH=fix(FS/85);
LOW=fix(FS/255);
for k=LOW:1:HIGH
    T=[T sum([zeros(1,k) S1 ].*[S1 zeros(1,k) ])];
end
[P,Q]=max(T);
pitchperiod=Q+LOW;
pitchfrequency=FS/pitchperiod;