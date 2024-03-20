clear;close all;clc


% "D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\"
% "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_echolalia"
Data_Folder ="Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_echolalia";
Fs = 16000;
flag =0;
flag_sift = 0;

Autism_data = dir(Data_Folder);

Param = struct();
% start with feature extraction
Param.alpha=0.98; % for pre emphasis
Param.WindowLength=30*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);

NumBands = 46;
range = [0,Fs/2];

[Filter_Bank,center_Frequencies,Filter_Bank_of_ones]...
  = Mel_Filter_bank(range,Param.WindowLenSamp,Fs,NumBands);



% Discrete cosine transform matrix..
[m,k] = meshgrid(0:NumBands-1);
m = m+1;   % m [1...M=numBands]

lamba_m = (2*m-1)/(2*NumBands);

% warped_lamba_m = th_p_of_Lamda1(alpha(1),lamba_m);
% DCT_mat = sqrt(2 / NumBands) * cos(pi * th_p_of_Lamda1(alpha(1),lamba_m).* k );
% DCT_mat(1,:) = DCT_mat(1,:) / sqrt(2);
DCT_mat = sqrt(2 / NumBands) * cos(pi *lamba_m.* k );
DCT_mat(1,:) = DCT_mat(1,:) / sqrt(2);
% round(DCT_mat*DCT_mat')


Data_Folder_in = Data_Folder+ "\" + Autism_data(7).name+ "\";

Autism_data_in = dir(Data_Folder_in);

% Autism_data(3).name
ADOS_table_path = Data_Folder+ "\" + Autism_data(7).name+ "\" + Autism_data(7).name +"_new.xlsx";

ADOS_table = readtable(ADOS_table_path);
ADOS_rec_path =Data_Folder+ "\" + Autism_data(7).name+ "\" + Autism_data(7).name +".wav";
%read the data

[Signal,~]=audioread(ADOS_rec_path);

[ProcessedSig,~] = PreProcess(...
  Signal,Fs,Param.alpha,Param.WindowLength,Param.Overlap);

ADOS_mat = table2array(ADOS_table(:,1:2));

StartTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3) | strcmp({'Therapist2'}, ADOS_table.Var3) ,1);
EndTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3)  | strcmp({'Therapist2'}, ADOS_table.Var3) ,2);

StartChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) | strcmp({'ChildEcholalia'}, ADOS_table.Var3) ,1);
EndChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) | strcmp({'ChildEcholalia'}, ADOS_table.Var3),2);

for i=1:length(StartTherapist)
  for j=1:length(StartChild)

    if ~(length(round(StartTherapist(i)*Fs):round(EndTherapist(i)*Fs))>= 110*10^(-3)*Fs &&...
        length(round(StartTherapist(i)*Fs):round(EndTherapist(i)*Fs))<= 3*Fs)
      continue
    else
      if ~(length(round(StartChild(i)*Fs):round(EndChild(i)*Fs))>= 110*10^(-3)*Fs &&...
          length(round(StartChild(i)*Fs):round(EndChild(i)*Fs))<= 3*Fs)
        continue
      else

        % start_C >= end_T and start_C - end_T <= 10:
        if EndTherapist(i) >= StartChild(j) && (StartChild(i) -EndTherapist(j)) >= 10

          dataT = ProcessedSig(round(StartTherapist(i)*Fs):round(EndTherapist(i)*Fs));

          speakreLabelT = ADOS_table.Var3(ADOS_table.Var1== StartTherapist(i));
          eventT = ADOS_table.Var4(ADOS_table.Var1== StartTherapist(i));
          start_timeT = ADOS_table.Var1(ADOS_table.Var1== StartTherapist(i));

          dataC = ProcessedSig(round(StartChild(i)*Fs):round(EndChild(i)*Fs));

          speakreLabelC = ADOS_table.Var3(ADOS_table.Var1== StartChild(i));
          eventC = ADOS_table.Var4(ADOS_table.Var1== StartChild(i));
          start_timeC = ADOS_table.Var1(ADOS_table.Var1== StartChild(i));

          [S,F,t] = stft(dataT,Fs, ...
            "Window",hamming(Param.WindowLenSamp), ...
            "OverlapLength",Param.noverlap, ...
            "FrequencyRange","onesided",...
            "FFTLength",Param.fftLength);
          PowerSpectrum_therapist = S.*conj(S);
          %           figure,plot(F,Filter_Bank.'),grid on;
          %         figure
          % surf(t,F,10*log10(PowerSpectrum_therapist),"EdgeColor","none");
          %         view([0,90])
          %         axis([t(1) t(end) F(1) F(end)])
          %         xlabel('Time (s)')
          %         ylabel('Frequency (Hz)')
          %         c = colorbar;
          %         c.Label.String = 'Power (dB)';

          FramedSigT = enframe(dataT ,hamming(Param.WindowLenSamp),Param.noverlap );
          LPC_matT = [];
          for w = 1:size(S,2)
            [~,LPc_dBT,~]=estimatePhonemeFormants(...
              FramedSigT(w,:),Fs,"h",0);
            LPC_matT = [LPC_matT LPc_dBT];

          end


          [SC,FC,tC] = stft(dataC,Fs, ...
            "Window",hamming(Param.WindowLenSamp), ...
            "OverlapLength",Param.noverlap, ...
            "FrequencyRange","onesided",...
            "FFTLength",Param.fftLength);
          PowerSpectrum_child = S.*conj(S);
          FramedSigC = enframe(dataC ,hamming(Param.WindowLenSamp),Param.noverlap );
          LPC_matC = [];
          for w = 1:size(SC,2)
            [~,LPc_dBC,~]=estimatePhonemeFormants(...
              FramedSigC(w,:),Fs,"h",0);
            LPC_matC = [LPC_matC LPc_dBC];

          end

          dist = dtw(LPC_matT,LPC_matC,20,"euclidean")



        end









      end


    end



  end
end

%
% if length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))>= 110*10^(-3)*Fs &&...
%             length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))<= 3*Fs
