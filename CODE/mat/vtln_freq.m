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
alpha = 1.1;
[frames_out_child] = Process_frame(frames_Child,Fs,Param,0,1,alpha);
[frames_out_Therapist] = Process_frame(frames_Therapist,Fs,Param,0,0,alpha);




pause(1)
soundsc(frames_out_Therapist(2).Signal_frame,16000);
figure(2);subplot(3,2,1);
 plot(frames_out_Therapist(2).t,frames_out_Therapist(2).Signal_frame)
title(['speech signal ' frames_out_Therapist(2).segment_event]);
xlabel('time[s]');ylabel('amp'); axis tight

figure(2);subplot(3,2,3)
plot(frames_out_child(2).t,frames_out_child(2).Signal_frame)
title(['speech signal ' frames_out_child(2).segment_event]);
xlabel('time[s]');ylabel('amp'); axis tight

T = 1/Fs;             % Sampling period
Len = length(frames_out_child(2).warpedSignal);
t = frames_out_child(2).t(1) + (0:Len-1)*T;% Time vector
figure(2);subplot(3,2,5); plot(t,frames_out_child(2).warpedSignal)
title(['speech signal warpedSignal ' frames_out_child(2).segment_event]);
xlabel('time[s]');ylabel('amp')
axis tight; hold on

figure(2);subplot(3,2,2)
surf(frames_out_Therapist(2).t_for_spect,frames_out_Therapist(2).F,...
    10*log10(frames_out_Therapist(2).PowerSpectrum),"DisplayName","P_{Spectrum}","EdgeColor","none");
view([0,90]); hold on
axis([frames_out_Therapist(2).t_for_spect(1) frames_out_Therapist(2).t_for_spect(end) frames_out_Therapist(2).F(1) frames_out_Therapist(2).F(end)])
xlabel('Time (s)');hold on
plot(frames_out_Therapist(2).t_for_spect,frames_out_Therapist(2).F1,"b","DisplayName","F_1");
hold on;
plot(frames_out_Therapist(2).t_for_spect,frames_out_Therapist(2).F2,"r","DisplayName","F_2");
hold on;
plot(frames_out_Therapist(2).t_for_spect,frames_out_Therapist(2).F3,"k","DisplayName","F_3");
hold on;ylabel('Frequency (Hz)');title("Therapist")

pause(2)
soundsc(frames_out_child(2).Signal_frame,16000);

figure(2);subplot(3,2,4)
surf(frames_out_child(2).t_for_spect,...
    frames_out_child(2).F,10*log10(frames_out_child(2).PowerSpectrum),...
    "DisplayName","Power Spectrum","EdgeColor","none");
view([0,90]); hold on
axis([frames_out_child(2).t_for_spect(1) frames_out_child(2).t_for_spect(end) frames_out_child(2).F(1) frames_out_child(2).F(end)])
xlabel('Time (s)');hold on
plot(frames_out_child(2).t_for_spect,frames_out_child(2).F1,"b","DisplayName","F_1");
hold on;
plot(frames_out_child(2).t_for_spect,frames_out_child(2).F2,"r","DisplayName","F_2");
hold on;
plot(frames_out_child(2).t_for_spect,frames_out_child(2).F3,"k","DisplayName","F_3");
hold on;
ylabel('Frequency (Hz)')
title("child")
pause(2)
soundsc(frames_out_child(2).warpedSignal,16000);

hold on; figure(2);subplot(3,2,6);
surf(frames_out_child(2).t_for_spectWarped,...
    frames_out_child(2).FWarped,10*log10(frames_out_child(2).PowerSpectrumWarped),...
    "DisplayName","Power Spectrum","EdgeColor","none");
view([0,90]); hold on
axis([frames_out_child(2).t_for_spectWarped(1) frames_out_child(2).t_for_spectWarped(end) frames_out_child(2).FWarped(1) frames_out_child(2).FWarped(end)])
xlabel('Time (s)');hold on
plot(frames_out_child(2).t_for_spectWarped,frames_out_child(2).F1Warped,"b","DisplayName","F_1");
hold on;
plot(frames_out_child(2).t_for_spectWarped,frames_out_child(2).F2Warped,"r","DisplayName","F_2");
hold on;
plot(frames_out_child(2).t_for_spectWarped,frames_out_child(2).F3Warped,"k","DisplayName","F_3");
hold on;
ylabel('Frequency (Hz)')
title("child Warped spectrum")