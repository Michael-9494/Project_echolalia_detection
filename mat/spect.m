
clear;close all;clc

%
%  gpuDevice(1)
%   done
%   Recs_for_cry_scream_25092022


Data_Folder = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_for_cry_scream_25092022\";
%   undone:
%   Recs_for_cry_scream_120822
%  Recs_for_cry_scream_18092022
%   Recs_for_cry_scream_010922
tic
% cd(Data_Folder);
%   664179718_131216\664179718_131216.wav
Autism_data = dir(Data_Folder);
all = struct();
Fs = 16000;
Param = struct();
% start with feature extraction
Param.alpha=0.98; % for pre emphasis
Param.WindowLength=20*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
% Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);

for file  = 3:length(Autism_data)

    % Autism_data(3).name
    ADOS_table_path = Data_Folder + Autism_data(file).name +"\" + Autism_data(file).name +"_new.xlsx";
    %
    % proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
    % ADOS_table_path = proj+"\New folder\675830557_170820_new.xlsx";
    ADOS_table = readtable(ADOS_table_path);
    ADOS_rec_path = Data_Folder + Autism_data(file).name +"\" + Autism_data(file).name +".wav";
    %read the data
    [Signal,~]=audioread(ADOS_rec_path);

    % PreProcess make all the pre processing wotk we need to do
    % the steps are based on DC removal, HP filtering.
    [ProcessedSig,~] = PreProcess(...
        Signal,Fs,Param.alpha,Param.WindowLength,Param.Overlap);

    ADOS_mat = table2array(ADOS_table(:,1:2));

    StartTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3) | strcmp({'Therapist2'}, ADOS_table.Var3) ,1);
    EndTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3)  | strcmp({'Therapist2'}, ADOS_table.Var3) ,2);

    StartChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) ,1);
    EndChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) ,2);

    frames_Therapist = splitWavByEvent(Signal, StartTherapist,EndTherapist,Fs,ADOS_table);
    %     all(file).frames_Therapist = all(file).frames_Therapist(...
    %         ~isempty( (all(file).frames_Therapist.start_time{1})  )  )  ;
    %     all(file).frames_Therapist = all(file).frames_Therapist(...
    %         ~cellfun(@isempty,struct2cell(all(file).frames_Therapist.event)));


    frames_Child = splitWavByEvent(Signal, StartChild,EndChild,Fs,ADOS_table);
    %     all(file).frames_Child = all(file).frames_Child(~isempty(all(file).frames_Child)) ;

    myFolders = split(Autism_data(file).name,"_");
    all(file).Child_name = myFolders{1};
    all(file).ADOS_date = myFolders{2};


    %     % find Therapist occurrences of echolalia from the data
    %     EcholaliaEventTherapistStart = ADOS_mat( (strcmp({'Echolalia'},...
    %         ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3)) |  ...
    %         (strcmp({'Echolalia'}, ADOS_table.Var4) & strcmp({'Therapist2'}, ADOS_table.Var3)),1);
    %
    %     EcholaliaEventTherapistEnd  = ADOS_mat( (strcmp({'Echolalia'},...
    %         ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3)) |  ...
    %         (strcmp({'Echolalia'}, ADOS_table.Var4) & strcmp({'Therapist2'}, ADOS_table.Var3)),2);
    %
    %     % find Child occurrences of echolalia from the data
    %     EcholaliaEventChildStart =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    %         & strcmp({'Child'}, ADOS_table.Var3) ,1);
    %     EcholaliaEventChildEnd =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    %         & strcmp({'Child'}, ADOS_table.Var3) ,2);
    %     if isempty(EcholaliaEventChildStart)
    %         continue
    %     end
    %     frames_Therapist = splitWavByEvent(ProcessedSig, EcholaliaEventTherapistStart,EcholaliaEventTherapistEnd,Fs,ADOS_table,Param);
    %     all(file).frames_Child = splitWavByEvent(ProcessedSig, EcholaliaEventChildStart,EcholaliaEventChildEnd,Fs,ADOS_table,Param);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % finished with reading child and therapist segments
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     clc


    %frames,Fs,Param,flag
    all(file).frames_out_Therapist = Process_frame(frames_Therapist,Fs,Param,0);
    all(file).frames_out_child = Process_frame(frames_Child,Fs,Param,0);


    all(file).F1_ther = [];all(file).F2_ther = [];all(file).F3_ther = [];%all(file).F4_ther = [];
    all(file).F1_ch = [];all(file).F2_ch = [];all(file).F3_ch = [];%F4_ch = [];
    for w = 1:length(all(file).frames_out_Therapist)
        all(file).F1_ther = [all(file).F1_ther all(file).frames_out_Therapist(w).F1];
        all(file).F2_ther = [all(file).F2_ther all(file).frames_out_Therapist(w).F2];
        all(file).F3_ther = [all(file).F3_ther all(file).frames_out_Therapist(w).F3];
        %         F4_ther = [F4_ther all(file).frames_out_Therapist(w).F4];


        % soundsc(frames_out_Ther(w).ProcessedSig,Fs);
        % soundsc(frames_out_ch(w).ProcessedSig,Fs);
    end
    all(file).mF1_th = mean(all(file).F1_ther(~isnan(all(file).F1_ther)));
    all(file).mF2_th = mean(all(file).F2_ther(~isnan(all(file).F2_ther)));
    all(file).mF3_th = mean(all(file).F3_ther(~isnan(all(file).F3_ther)));
    %     mF4_th(file) = mean(F4_ther(~isnan(F4_ther)));

    for w = 1:length(all(file).frames_out_child)

        all(file).F1_ch = [all(file).F1_ch all(file).frames_out_child(w).F1];
        all(file).F2_ch = [all(file).F2_ch all(file).frames_out_child(w).F2];
        all(file).F3_ch = [all(file).F3_ch all(file).frames_out_child(w).F3];
        %         all(file).F4_ch = [all(file).F4_ch all(file).frames_out_child(w).F4];
    end
    all(file).mF1_ch = mean(all(file).F1_ch(~isnan(all(file).F1_ch)));
    all(file).mF2_ch = mean(all(file).F2_ch(~isnan(all(file).F2_ch)));
    all(file).mF3_ch = mean(all(file).F3_ch(~isnan(all(file).F3_ch)));
    %     all(file).mF4_ch = mean(all(file).F4_ch(~isnan(all(file).F4_ch)));
    Process_time_one_recording = toc;
    % show time elapsed
    hours = floor(Process_time_one_recording/3600);
    minuts = floor(Process_time_one_recording/60) - hours * 60;
    seconds = floor(Process_time_one_recording - hours * 3600  - minuts * 60);

    sprintf("patient num %s from date %s is done."...
        ,myFolders{1},myFolders{2})
    disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);

end
all = all(3:end);
tic
save('Recs_for_cry_scream_25092022.mat','all', '-v7.3')

% show time elapsed
hours = floor(toc/3600);
minuts = floor(toc/60) - hours * 60;
seconds = floor(toc - hours * 3600  - minuts * 60);
disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
disp("Recs_for_cry_scream_25092022 done and saved")
