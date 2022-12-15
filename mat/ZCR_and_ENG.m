function [Frames_with_vocal_phoneme,Signal_Energy,ZeroCrossingSignal] = ZCR_and_ENG(FramedSig,Eb_T,Eb_C,ZC_T,ZC_C,segment_speaker)
% ZCR_and_ENG finds the start and end of speech using energy
% calculation. based of certain threshold
% FramedSig – the framed speech signal after preprocessing.
% Fs – sampling frequency
% WindowLength – length of test and reference windows [seconds]
% Overlap – percentage of overlap between adjacent frames [0-100]

% OUTPUT:
% Idx – 2 integer vector: start and end indices of detected word.
% %
Signal_Energy=calcNRG(FramedSig);%calculate  of each frame
ZeroCrossingSignal = calcZCR(FramedSig);
% Frames_with_vocal_phoneme = zeros([1 length(Signal_Energy)]);
% mzcr = mean(ZeroCrossingSignal);
% vzcr = var(ZeroCrossingSignal,1);
% mnrg = mean(Signal_Energy);
% vnrg = var(Signal_Energy,1);
% % Frames_with_vocal_phoneme=find(Signal_Energy>mnrg & ZeroCrossingSignal<mzcr);%
% 
% % % baseline energy level, Eb
% %
% EbNRG = mnrg-(sqrt(vnrg)/2);%;
% EbZCR = mzcr+(sqrt(vzcr)/2);%;
% Frames_with_vocal_phoneme=find(Signal_Energy>=EbNRG & ZeroCrossingSignal<=EbZCR);%
% % 
% Eb_T = Param.NRG_median_therapist;
% Eb_C =  Param.NRG_median_Child;
% 
% 
% ZC_T =Param.ZCR_median_therapist;
% ZC_C =  Param.ZCR_median_Child;

if strcmp(segment_speaker,"Therapist")
    Frames_with_vocal_phoneme=find(Signal_Energy>=Eb_T & ZeroCrossingSignal<=ZC_T);%
% %     figure
% %     subplot(2,1,1)
% %     plot(ZeroCrossingSignal,"DisplayName","ZCR"),xlabel('frame number');
% %     ylabel('Zero-crossing rate'),title("Signal ZCR");yline(ZC_T,'--m',"DisplayName","ZC therapist");
% %     subplot(2,1,2),plot(1:length(Signal_Energy),Signal_Energy,"--o","DisplayName","NRG");
% %     title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
% %     grid on; hold on
% %     yline(Eb_T,'--m',"DisplayName","NRG therapist");%yline(th1,'b');yline(th2,'g')
% %     legend();
% %     hold on
elseif strcmp(segment_speaker,"Child")
    Frames_with_vocal_phoneme=find(Signal_Energy>=Eb_C & ZeroCrossingSignal<=ZC_C);%
% %         figure
% %     subplot(2,1,1)
% %     plot(ZeroCrossingSignal,"DisplayName","ZCR"),xlabel('frame number');
% %     ylabel('Zero-crossing rate'),title("Signal ZCR");
% %     yline(ZC_C,'--m',"DisplayName","ZC child");
% %     subplot(2,1,2),plot(1:length(Signal_Energy),Signal_Energy,"--o","DisplayName","NRG");
% %     title("Signal Energy"); xlabel 'segment index' ; ylabel 'Energy [dB]'
% %     grid on; hold on
% %     yline(Eb_C,'--m',"DisplayName","NRG child");
% %     legend();
end
% th2 = 10 * log10(1.1) + EbNRG;
% Frames_without_vocal_phoneme=find(Signal_Energy<th2);
%
% WindowLength_samples=WindowLength*Fs;   % [sec]*[sample/sec]=[sample]
% overlap_in_samples=((Overlap)*WindowLength_samples)/100; % overlap in samples






% Idx=[];
% Seg_end = Frames_with_vocal_phoneme+1;
% Seg_end = Seg_end(2:end);
% Seg_end = [Seg_end ;Frames_with_vocal_phoneme(end)+1];
% idx_vec_end = Seg_end*overlap_in_samples;

% Seg_start = Frames_without_vocal_phoneme;
% idx_vec_start = Seg_start*overlap_in_samples;


% % find relevant index acording to the overlaping.
% idx_start=(Frames_with_vocal_phoneme(1))*overlap_in_samples;
% Idx(1)=idx_start;
% idx_end=(Frames_with_vocal_phoneme(end))*overlap_in_samples;
% Idx(2)=idx_end;%+WindowLength_samples-1;

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



