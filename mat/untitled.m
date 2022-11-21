
clear;close all;clc
% 
% Data_Folder = "Y:\Echolalia_proj_Michael\DATA\New folder";
% % cd(Data_Folder);
% 
% Autism_data = dir(Data_Folder);

proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
ADOS_table = readtable(proj+"\New folder\675830557_170820_new.xlsx");

[Signal,Fs]=audioread(proj+"\New folder\675830557_170820.wav");

ADOS_mat = table2array(ADOS_table(:,1:2));

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

% lets look at the first occurrence of echolalia at the therapist
Signal_therapist = Signal(EcholaliaEventTherapistStart(1)*Fs:EcholaliaEventTherapistEnd(1)*Fs);
Signal_child = Signal(EcholaliaEventChildStart(1)*Fs:EcholaliaEventChildEnd(1)*Fs);

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



% soundsc(Signal_therapist,Fs);
% soundsc(Signal_child,Fs);

% numHops = floor((r-winLength)/hopLength) + 1
alpha=0.98;
WindowLength=40*10^-3;  % 30 [mS] window
WindowLenSamp=WindowLength*Fs;
Overlap=75;             % 50% overlap
fftLength = 2^nextpow2(WindowLenSamp);
noverlap=((Overlap)*WindowLength)/100 *Fs;


%  we defined a baseline energy level, Eb, (in dB),
% as the most frequent energy level (i.e., background noise)
% within the audio interval and its vicinity (+- 20 s).

[ProcessedSig,FramedSig] = PreProcess(Signal(...
    EcholaliaEventTherapistStart(1)*Fs-5*Fs:EcholaliaEventTherapistEnd(1)*Fs+5*Fs),Fs,alpha,WindowLength,Overlap);
Signal_Energy=calcNRG(FramedSig);

frames_Therapist = splitWavByEvent(Signal, EcholaliaEventTherapistStart,EcholaliaEventTherapistEnd,Fs);
frames_Child = splitWavByEvent(Signal, EcholaliaEventChildStart,EcholaliaEventChildEnd,Fs);

% baseline energy level, Eb
Eb = mean(Signal_Energy);

[ProcessedSig_therapist,FramedSig_therapist] = PreProcess(Signal_therapist,Fs,alpha,WindowLength,Overlap);
[ProcessedSig_child,FramedSig_child] = PreProcess(Signal_child,Fs,alpha,WindowLength,Overlap);
% soundsc(ProcessedSig_therapist,Fs);
% soundsc(ProcessedSig_child,Fs);
%  [Idx,idx_vec_start,idx_vec_end] = FindWordIdx(ProcessedSig_therapist,FramedSig_therapist,Fs,WindowLength,Overlap);


