function [res1,res2,speechsegment,utforste,ltforste,ltforzcr]...
    =endpointdetection(Sig,FS,WindowLenSamp)
%mzcr, mste-mean of the zero-crossing rate and the
% short-time energy for the first 100\,ms
%vzcr,vste-variance of the zero-crossing rate and the
% short-time energy for the first 100ms
%utforste-upper threshold for short-time energy
%ltforste-lower threshold for short-time energy
%ltforzcr-lower threshold for zero-crossing rate
%fl - frame length is fixed as 10 ms
fl=WindowLenSamp;
Sig = Sig';
%DC offset removal
% Sig=(Sig-mean(Sig))';
tempdata=Sig(1:1:fl*20);
% zcr = @(x) length(find(diff(cumsum(sign(x)))==-1));
ste = @(block_struct) sum(block_struct.data.^2)/length(block_struct.data);
zcr = @(block_struct) length(find(diff(cumsum(sign(block_struct.data)))==-1));

temp1=blockproc(tempdata,[1 fl],zcr);
mzcr=mean(temp1);
vzcr=var(temp1,1);
temp2=blockproc(tempdata,[1 fl],ste);
mste=mean(temp2);
vste=var(temp2,1);
ltforste=mste*100-(sqrt(vste)/10);
utforste=mste*100+(sqrt(vste)/10);
ltforzcr=mzcr*100-(sqrt(vzcr)/10);
res1=blockproc(Sig,[1 fl],zcr);
res2=blockproc(Sig,[1 fl],ste);
figure(1)
subplot(2,1,1)
plot(res1)
xlabel('frame number')
ylabel('Zero-crossing rate')
subplot(2,1,2)
plot(res2)
xlabel('frame number')
ylabel('Short-time energy')


[p1,q1]=find(res2>utforste);
temp3=res2(q1(1):-1:1);
[p2,q2]=find(temp3<ltforste);
if(isempty(q2)==1)
    q2(1)=0;
    temp4=res1(q1(1)-q2(1):-1:1);
else
    temp4=res1(q1(1)-q2(1):-1:1);
end
[p3,q3]=find(temp4<ltforzcr);
res2rev=res2(length(res2):-1:1);
[p4,q4]=find(res2rev>utforste);
temp5=res2rev(q4(1):-1:1);
[p5,q5]=find(temp5<ltforste);
res1rev=res1(length(res1):-1:1);
if(isempty(q5)==1)
    q5(1)=0;
    temp6=res1rev(q4(1)-q5(1):-1:1);
else
    temp6=res1rev(q4(1)-q5(1):-1:1);
end
[p6,q6]=find(temp6<ltforzcr);
speechsegment=Sig((length(temp4)-q3(1)+1)*fl:1:length(Sig)...
    -(length(temp6)-q6(1)+1)*fl);
figure
subplot(2,1,1)
plot(Sig)
title('Original speech signal')
subplot(2,1,2)
plot(speechsegment)
title('Speech segment after endpoint detection')
end