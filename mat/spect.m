
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

all = struct();
for i = 1:length(Recs_for_cry_scream1)

    Data_Folder = Data_Folder_out +"Recs_for_cry_scream"+ Recs_for_cry_scream1(i)+ "\";

    tic

    Autism_data_in = dir(Data_Folder);
    clear all ;
    all = struct();


    for file  = 3:length(Autism_data_in)

        % Autism_data(3).name
        ADOS_table_path = Data_Folder + Autism_data_in(file).name +"\" + Autism_data_in(file).name +"_new.xlsx";
        %
        % proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
        % ADOS_table_path = proj+"\New folder\675830557_170820_new.xlsx";
        try
            ADOS_table = readtable(ADOS_table_path);
        catch ME
            fprintf('readtable without success: %s\n', ME.message);
            continue;  % Jump to next iteration of: for i
        end

        ADOS_rec_path = Data_Folder + Autism_data_in(file).name +"\" + Autism_data_in(file).name +".wav";
        %read the data
        try
            [Signal,~]=audioread(ADOS_rec_path);
        catch ME
            fprintf('audioread without success: %s\n', ME.message);
            continue;  % Jump to next iteration of: for i
        end


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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % finished with reading child and therapist segments
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        myFolders = split(Autism_data_in(file).name,"_");
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


        for w = 1:length(all(file).frames_out_child)

            all(file).F1_ch = [all(file).F1_ch all(file).frames_out_child(w).F1];
            all(file).F2_ch = [all(file).F2_ch all(file).frames_out_child(w).F2];
            all(file).F3_ch = [all(file).F3_ch all(file).frames_out_child(w).F3];
            %         all(file).F4_ch = [all(file).F4_ch all(file).frames_out_child(w).F4];
        end

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
    save("Recs_for_cry_scream1"+Recs_for_cry_scream1(i)+".mat",'all', '-v7.3')

    % show time elapsed
    hours = floor(toc/3600);
    minuts = floor(toc/60) - hours * 60;
    seconds = floor(toc - hours * 3600  - minuts * 60);
    disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
    disp("Recs_for_cry_scream"+Recs_for_cry_scream1(i)+".mat done and saved")

end