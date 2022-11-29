clear;close all;clc
Data_Folder_out = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\";
Fs = 16000;
flag =0;
Autism_data = dir(Data_Folder_out); Autism_data = Autism_data(4:end);
Recs_for_cry_scream =[Autism_data(1).name, Autism_data(2).name, Autism_data(3).name,...
    Autism_data(4).name, Autism_data(5).name, Autism_data(6).name,...
    Autism_data(7).name, Autism_data(8).name, Autism_data(9).name];
Recs_for_cry_scream1 = split(Recs_for_cry_scream,"Recs_for_cry_scream");
Recs_for_cry_scream1 = Recs_for_cry_scream1(2:end);

Param = struct();
% start with feature extraction
Param.alpha=0.98; % for pre emphasis
Param.WindowLength=20*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
% Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);



Data_Folder = Data_Folder_out +"Recs_for_cry_scream"+ Recs_for_cry_scream1(7)+ "\";

Autism_data_in = dir(Data_Folder);

% Autism_data(3).name
ADOS_table_path = Data_Folder + Autism_data_in(7).name +"\" + Autism_data_in(7).name +"_new.xlsx";
%
% proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
% ADOS_table_path = proj+"\New folder\675830557_170820_new.xlsx";
try
    ADOS_table = readtable(ADOS_table_path);
catch ME
    fprintf('readtable without success: %s\n', ME.message);
    % Jump to next iteration of: for i
end

ADOS_rec_path = Data_Folder + Autism_data_in(7).name +"\" + Autism_data_in(7).name +".wav";
%read the data
try
    [Signal,~]=audioread(ADOS_rec_path);
catch ME
    fprintf('audioread without success: %s\n', ME.message);
    % Jump to next iteration of: for i
end



ADOS_mat = table2array(ADOS_table(:,1:2));
[ProcessedSig,~] = PreProcess(...
    Signal,Fs,Param.alpha,Param.WindowLength,Param.Overlap);

EcholaliaEventTherapistStart = ADOS_mat( (strcmp({'Echolalia'},...
    ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3)) |  ...
    (strcmp({'Echolalia'}, ADOS_table.Var4) & strcmp({'Therapist2'}, ADOS_table.Var3)),1);

EcholaliaEventTherapistEnd  = ADOS_mat( (strcmp({'Echolalia'},...
    ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3)) |  ...
    (strcmp({'Echolalia'}, ADOS_table.Var4) & strcmp({'Therapist2'}, ADOS_table.Var3)),2);

% find Child occurrences of echolalia from the data
EcholaliaEventChildStart =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    & strcmp({'Child'}, ADOS_table.Var3) ,1);
EcholaliaEventChildEnd =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    & strcmp({'Child'}, ADOS_table.Var3) ,2);

frames_Therapist = splitWavByEvent(ProcessedSig, EcholaliaEventTherapistStart,EcholaliaEventTherapistEnd,Fs,ADOS_table,Param);
frames_Child = splitWavByEvent(ProcessedSig, EcholaliaEventChildStart,EcholaliaEventChildEnd,Fs,ADOS_table,Param);
[frames_out_Child] = Process_frame(frames_Child,Fs,Param,0)

alpha=1.2;
fRatio=1.2;
freqs_Child = time2freq(frames_out_Child);
warpedFreqs = vtln(freqs_Child, "symmetric", alpha);

warpedWav = freq2time(warpedFreqs);
wavOut = psola(warpedWav, fRatio);
audiowrite(string(Autism_data_in(3).name +"_29112022.wav"),wavOut, Fs );
