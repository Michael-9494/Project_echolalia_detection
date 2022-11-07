function [Idx,idx_vec_start,idx_vec_end] = FindWordIdx(FramedSig,Fs,WindowLength,Overlap)
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
% baseline energy level, Eb
Eb = mean(Signal_Energy);
% thresh=max(Signal_Energy)*0.02;

th1 = 10 * log10(1.9) + Eb;
Frames_with_Speech=find(Signal_Energy>th1);
th2 = 10 * log10(1.1) + Eb;
Frames_without_Speech=find(Signal_Energy<th2);



WindowLength_samples=WindowLength*Fs;   % [sec]*[sample/sec]=[sample]
overlap_in_samples=((Overlap)*WindowLength_samples)/100; % overlap in samples


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
end


