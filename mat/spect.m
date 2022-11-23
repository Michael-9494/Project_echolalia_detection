
clear;close all;clc
% 
% Data_Folder = "Y:\Echolalia_proj_Michael\DATA\New folder";
% % cd(Data_Folder);
% 
% Autism_data = dir(Data_Folder);

proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
ADOS_table_path = proj+"\New folder\675830557_170820_new.xlsx";
ADOS_table = readtable(ADOS_table_path);
ADOS_rec_path = proj+"\New folder\675830557_170820.wav";
%read the data 
[Signal,Fs]=audioread(ADOS_rec_path);
ADOS_mat = table2array(ADOS_table(:,1:2));

StartTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3) ,1);
EndTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3) ,2);

StartChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) ,1);
EndChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) ,2);

frames_Therapist = splitWavByEvent(Signal, StartTherapist,EndTherapist,Fs,ADOS_table);
frames_Child = splitWavByEvent(Signal, StartChild,EndChild,Fs,ADOS_table);



% find Therapist occurrences of echolalia from the data
EcholaliaEventTherapistStart = ADOS_mat( strcmp({'Echolalia'},...
    ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3) ,1);
EcholaliaEventTherapistEnd =ADOS_mat( strcmp({'Echolalia'},...
    ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3) ,2);

% find Child occurrences of echolalia from the data
EcholaliaEventChildStart =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    & strcmp({'Child'}, ADOS_table.Var3) ,1);
EcholaliaEventChildEnd =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    & strcmp({'Child'}, ADOS_table.Var3) ,2);

Echo_frames_Therapist = splitWavByEvent(Signal, EcholaliaEventTherapistStart,EcholaliaEventTherapistEnd,Fs,ADOS_table);
Echo_frames_Child = splitWavByEvent(Signal, EcholaliaEventChildStart,EcholaliaEventChildEnd,Fs,ADOS_table);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% finished with reading child and therapist segments 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
Param = struct();
% start with feature extraction
Param.alpha=0.98; % for pre emphasis
Param.WindowLength=30*10^-3;  % 30 [mS] window
Param.WindowLenSamp=Param.WindowLength*Fs;
Param.Overlap=50;             % 50% overlap
% Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=((Param.Overlap)*Param.WindowLength)/100 *Fs;

[frames_out] = Process_frame(Echo_frames_Therapist,Fs,Param,0);
[frames_out_Therapist] = Process_frame(frames_Therapist,Fs,Param,0);

%%
% lets look at the first occurrence of echolalia at the therapist
Signal_therapist = Echo_frames_Therapist(1).data;
Signal_child = Echo_frames_Child(1).data;
% Default values for the tracking with voiced/unvoiced decision

T = 1/Fs;             % Sampling period
Len_thera = length(Signal_therapist);
Len_child = length(Signal_child);

tTherapist = EcholaliaEventTherapistStart(1)+(0:Len_thera-1)*T;% Time vector
tChild = EcholaliaEventChildStart(1)+(0:Len_child-1)*T;

figure,subplot(2,1,1)
plot(tTherapist,Signal_therapist)
title('speech signal therapist');xlabel('time[s]');ylabel('amp')
grid on; axis tight
hold on
subplot(2,1,2)
plot(tChild,Signal_child)
title('speech signal child');xlabel('time[s]');ylabel('amp')
grid on; axis tight

% soundsc(Signal_therapist(1:0.2*Fs),Fs);
% soundsc(Signal_child(1:0.2*Fs),Fs);


[ProcessedSig_therapist,FramedSig_therapist] = PreProcess(Signal_therapist,Fs,alpha,WindowLength,Overlap);
[ProcessedSig_child,FramedSig_child] = PreProcess(Signal_child,Fs,alpha,WindowLength,Overlap);

% Convert the audio signal to a frequency-domain representation using 30 ms
% windows with 15 ms overlap. Because the input is real and therefore the
% spectrum is symmetric, you can use just one side of the frequency domain
% representation without any loss of information. Convert the complex
% spectrum to the magnitude spectrum: phase information is discarded
% when calculating mel frequency cepstral coefficients (MFCC).


[S,F,t] = stft(ProcessedSig_therapist,Fs, ...
    "Window",hamming(WindowLenSamp,"periodic"), ...
    "OverlapLength",noverlap, ...
    "FrequencyRange","onesided");
PowerSpectrum_therapist = S.*conj(S);


figure
surf(t,F,10*log10(PowerSpectrum_therapist),"EdgeColor","none");
view([0,90])
axis([t(1) t(end) F(1) F(end)])
xlabel('Time (s)')
ylabel('Frequency (Hz)')
c = colorbar;
c.Label.String = 'Power (dB)';
title("Therapist")



[Idx,idx_vec_start,idx_vec_end] = FindWordIdx(ProcessedSig_therapist,FramedSig_therapist,Fs,WindowLength,Overlap);
[h1,h2]=estimatePhonemeFormants(ProcessedSig_therapist(1:0.2*Fs),Fs,"h");
[Idx2,idx_vec_start2,idx_vec_end2] = FindWordIdx(ProcessedSig_child,FramedSig_child,Fs,WindowLength,Overlap);
[h12,h22]=estimatePhonemeFormants(ProcessedSig_child(1:0.2*Fs),Fs,"h");
