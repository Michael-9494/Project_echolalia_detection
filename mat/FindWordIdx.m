function [Idx,idx_vec_start,idx_vec_end] = FindWordIdx(ProcessedSig,FramedSig,Fs,WindowLength,Overlap)
% FindWordIdx finds the start and end of speech using energy
% calculation. based of certain threshold
% FramedSig – the framed speech signal after preprocessing.
% Fs – sampling frequency
% WindowLength – length of test and reference windows [seconds]
% Overlap – percentage of overlap between adjacent frames [0-100]

% OUTPUT:
% Idx – 2 integer vector: start and end indices of detected word.
%
Signal_Energy=calcNRG(FramedSig);%calculate  of each frame
ZeroCrossingSignal = calcZCR(FramedSig);


% baseline energy level, Eb
Eb = mean(Signal_Energy);
thresh=max(Signal_Energy)*0.02;

th1 = 10 * log10(1.9) + Eb;
Frames_with_Speech=find(Signal_Energy>th1);
th2 = 10 * log10(1.1) + Eb;
Frames_without_Speech=find(Signal_Energy<th2);

WindowLength_samples=WindowLength*Fs;   % [sec]*[sample/sec]=[sample]
overlap_in_samples=((Overlap)*WindowLength_samples)/100; % overlap in samples


figure
subplot(2,1,1)
plot(ZeroCrossingSignal)
xlabel('frame number')
ylabel('Zero-crossing rate')
subplot(2,1,2)
plot(1:length(Signal_Energy),Signal_Energy,"--o"); title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
grid on; hold on
yline(Eb,'--m');yline(th1,'b');yline(th2,'g')
legend('Energy','Eb ','th1','th2');
hold on



Idx=[];
Seg_end = Frames_with_Speech((diff(Frames_with_Speech)>17))+1;
% Seg_end = Seg_end(2:end);
Seg_end = [Seg_end ;Frames_with_Speech(end)+1];
idx_vec_end = Seg_end*overlap_in_samples;

Seg_start = Frames_without_Speech((diff(Frames_without_Speech)>17));
idx_vec_start = Seg_start*overlap_in_samples;


% find relevant index acording to the overlaping.
idx_start=(Frames_with_Speech(1))*overlap_in_samples;
Idx(1)=idx_start;
idx_end=(Frames_with_Speech(end))*overlap_in_samples;
Idx(2)=idx_end;%+WindowLength_samples-1;


figure
plot(1:length(Signal_Energy),Signal_Energy,"--o"); title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
grid on; hold on
yline(Eb,'--m');yline(th1,'b');yline(th2,'g')
legend('Energy','Eb ','th1','th2');
hold on
% for i=1:length(idx_vec_start)
line([(Seg_start) (Seg_start)],ylim,'color','r','linewidth',1.2)
hold on
line([(Seg_end) (Seg_end)],ylim,'color','b','linewidth',1.2)
% end
%  line([(Seg_end(end)) (Seg_end(end))],ylim,'color','b','linewidth',1.2)
legend('speech','Eb','th1','th2','start ','end');grid on


% 
% 
% temp1=ZeroCrossingSignal(1:10);
% mzcr=mean(temp1);
% vzcr=var(temp1,1);
% temp2=Signal_Energy(1:10);
% mste=mean(temp2);
% vste=var(temp2,1);
% ltforste=mste*100-(sqrt(vste)/10);
% utforste=mste*100+(sqrt(vste)/10);
% ltforzcr=mzcr*100-(sqrt(vzcr)/10);
% res1=ZeroCrossingSignal;
% res2=Signal_Energy;
% figure
% subplot(2,1,1)
% plot(res1)
% xlabel('frame number')
% ylabel('Zero-crossing rate')
% subplot(2,1,2)
% plot(res2)
% xlabel('frame number')
% ylabel('Short-time energy')
% 
% 
% [p1,q1]=find(res2>utforste);
% temp3=res2(q1(1):-1:1);
% [p2,q2]=find(temp3<ltforste);
% if(isempty(q2)==1)
%     q2(1)=0;
%     temp4=res1(q1(1)-q2(1):-1:1);
% else
%     temp4=res1(q1(1)-q2(1):-1:1);
% end
% [p3,q3]=find(temp4<ltforzcr);
% res2rev=res2(length(res2):-1:1);
% [p4,q4]=find(res2rev>utforste);
% temp5=res2rev(q4(1):-1:1);
% [p5,q5]=find(temp5<ltforste);
% res1rev=res1(length(res1):-1:1);
% if(isempty(q5)==1)
%     q5(1)=0;
%     temp6=res1rev(q4(1)-q5(1):-1:1);
% else
%     temp6=res1rev(q4(1)-q5(1):-1:1);
% end
% [p6,q6]=find(temp6<ltforzcr);
% speechsegment=ProcessedSig((length(temp4)-q3(1)+1)*WindowLength:1:length(ProcessedSig)...
%     -(length(temp6)-q6(1)+1)*WindowLength);
% figure
% subplot(2,1,1)
% plot(ProcessedSig)
% title('Original speech signal')
% subplot(2,1,2)
% plot(speechsegment)
% title('Speech segment after endpoint detection')
end


