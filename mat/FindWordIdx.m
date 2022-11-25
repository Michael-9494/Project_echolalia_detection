function [Idx,Frames_with_vocal_phoneme] = FindWordIdx(FramedSig,Fs,WindowLength,Overlap)
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
EbNRG = mean(Signal_Energy);
EbZCR = mean(ZeroCrossingSignal);
thresh=max(Signal_Energy)*0.02;

th1 = 10 * log10(1.9) + EbNRG;
Frames_with_vocal_phoneme=find(Signal_Energy>EbNRG & Signal_Energy<EbZCR);% 
th2 = 10 * log10(1.1) + EbNRG;
Frames_without_vocal_phoneme=find(Signal_Energy<th2);

WindowLength_samples=WindowLength*Fs;   % [sec]*[sample/sec]=[sample]
overlap_in_samples=((Overlap)*WindowLength_samples)/100; % overlap in samples


% figure
% subplot(2,1,1)
% plot(ZeroCrossingSignal),xlabel('frame number');
% ylabel('Zero-crossing rate'),title("Signal ZCR");yline(EbZCR,'--m');
% subplot(2,1,2),plot(1:length(Signal_Energy),Signal_Energy,"--o");
% title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
% grid on; hold on
% yline(EbNRG,'--m');yline(th1,'b');yline(th2,'g')
% legend('Energy','Eb ','th1','th2');
% hold on



Idx=[];
% Seg_end = Frames_with_vocal_phoneme+1;
% Seg_end = Seg_end(2:end);
% Seg_end = [Seg_end ;Frames_with_vocal_phoneme(end)+1];
% idx_vec_end = Seg_end*overlap_in_samples;

% Seg_start = Frames_without_vocal_phoneme;
% idx_vec_start = Seg_start*overlap_in_samples;


% find relevant index acording to the overlaping.
idx_start=(Frames_with_vocal_phoneme(1))*overlap_in_samples;
Idx(1)=idx_start;
idx_end=(Frames_with_vocal_phoneme(end))*overlap_in_samples;
Idx(2)=idx_end;%+WindowLength_samples-1;

% 
% figure
% plot(1:length(Signal_Energy),Signal_Energy,"--o"); title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
% grid on; hold on
% yline(Eb,'--m');yline(th1,'b');yline(th2,'g')
% legend('Energy','Eb ','th1','th2');
% hold on
% for i=1:length(idx_vec_start)
% line([(Seg_start) (Seg_start)],ylim,'color','r','linewidth',1.2)

% line([(Seg_end) (Seg_end)],ylim,'color','b','linewidth',1.2)
% end
%  line([(Seg_end(end)) (Seg_end(end))],ylim,'color','b','linewidth',1.2)
% legend('speech','Eb','th1','th2','start ','end');grid on


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
% [m,n] = size(FramedSig); %number of windows of signal
% word_win = zeros(m,1); %intializing the window flag on energy
% 
% for i = 1:m
% signal_F = fft(FramedSig(i,:));
% power_win = sum(signal_F.*conj(signal_F));
% 
%     if power_win > (mean(sum(abs(FramedSig)))*4.5)
%        word_win(i) = 1;
%     end
% end
% 
% num_window_word = find(word_win == 1); %finding the window numbe of last and end of word
% begin_win = num_window_word(1);
% end_win = num_window_word(end);
% overlap_in_samples=((Overlap)*WindowLength_samples)/100; % overlap in samples

% begin_index_first_win = ((n*Overlap)/100)*(begin_win-1)+1; %inexs in original signal of begin and end of word
% end_index_last_win = ((n*Overlap)/100)*(end_win);
% 
% Idx = [begin_index_first_win,end_index_last_win];
% 
% end
end


