
clear;close all;clc

%
%  gpuDevice(1)
%   done
%   Recs_for_cry_scream_25092022    
%   Recs_for_cry_scream_18092022

Data_Folder = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_for_cry_scream_010922\";
%   undone:
%   

% cd(Data_Folder);
%   664179718_131216\664179718_131216.wav
Autism_data = dir(Data_Folder);
all = struct();
for file  = 3:length(Autism_data)
    tic
    % Autism_data(3).name
    ADOS_table_path = Data_Folder + Autism_data(file).name +"\" + Autism_data(file).name +"_new.xlsx";
    %
    % proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
    % ADOS_table_path = proj+"\New folder\675830557_170820_new.xlsx";
    ADOS_table = readtable(ADOS_table_path);
    ADOS_rec_path = Data_Folder + Autism_data(file).name +"\" + Autism_data(file).name +".wav";
    %read the data
    [Signal,Fs]=audioread(ADOS_rec_path);
    ADOS_mat = table2array(ADOS_table(:,1:2));

    StartTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3) ,1);
    EndTherapist = ADOS_mat( strcmp({'Therapist'}, ADOS_table.Var3) ,2);

    StartChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) ,1);
    EndChild = ADOS_mat( strcmp({'Child'}, ADOS_table.Var3) ,2);

    all(file).frames_Therapist = splitWavByEvent(Signal, StartTherapist,EndTherapist,Fs,ADOS_table);
    all(file).frames_Child = splitWavByEvent(Signal, StartChild,EndChild,Fs,ADOS_table);



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

%     Echo_frames_Therapist = splitWavByEvent(Signal, EcholaliaEventTherapistStart,EcholaliaEventTherapistEnd,Fs,ADOS_table);
%     Echo_frames_Child = splitWavByEvent(Signal, EcholaliaEventChildStart,EcholaliaEventChildEnd,Fs,ADOS_table);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % finished with reading child and therapist segments
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clc
    Param = struct();
    % start with feature extraction
    Param.alpha=0.96; % for pre emphasis
    Param.WindowLength=30*10^-3;  % 30 [mS] window
    Param.WindowLenSamp=Param.WindowLength*Fs;
    Param.Overlap=50;             % 50% overlap
    % Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
    Param.noverlap=((Param.Overlap)*Param.WindowLength)/100 *Fs;

%     [frames_out_Ther] = Process_frame(Echo_frames_Therapist,Fs,Param,0);
%     [frames_out_ch] = Process_frame(Echo_frames_Child,Fs,Param,0);
    %
    [all(file).frames_out_Therapist] = Process_frame(all(file).frames_Therapist,Fs,Param,0);
    [all(file).frames_out_child] = Process_frame(all(file).frames_Child,Fs,Param,0);


    F1_ther = [];F2_ther = [];F3_ther = [];F4_ther = [];
    F1_ch = [];F2_ch = [];F3_ch = [];F4_ch = [];
    for w = 1:length(all(file).frames_out_Therapist)
        %     figure,subplot(2,2,1)
        %     plot(frames_out_Ther(w).t,frames_out_Ther(w).ProcessedSig)
        %     title('speech signal therapist');xlabel('time[s]');ylabel('amp')
        %     grid on; axis tight
        %
        %     subplot(2,2,2)
        %     surf(frames_out_Ther(w).t_for_spect,frames_out_Ther(w).F,10*log10(frames_out_Ther(w).PowerSpectrum),...
        %         "DisplayName","P_{Spectrum}","EdgeColor","none");
        %     view([0,90]); hold on
        %     axis([frames_out_Ther(w).t_for_spect(1) frames_out_Ther(w).t_for_spect(end) frames_out_Ther(w).F(1) frames_out_Ther(w).F(end)])
        %     xlabel('Time (s)');hold on
        %     plot(frames_out_Ther(w).t_for_spect,frames_out_Ther(w).F1,"b","DisplayName","F_1");
        %     hold on;
        %     plot(frames_out_Ther(w).t_for_spect,frames_out_Ther(w).F2,"r","DisplayName","F_2");
        %     hold on;
        %     plot(frames_out_Ther(w).t_for_spect,frames_out_Ther(w).F3,"k","DisplayName","F_3");
        %     hold on;
        %     plot(frames_out_Ther(w).t_for_spect,frames_out_Ther(w).F4,"g","DisplayName","F_4");
        %     ylabel('Frequency (Hz)')
        %     c = colorbar;
        %     c.Label.String = 'Power (dB)';
        %     title("Therapist")
        %     legend
        %     hold on; subplot(2,2,3); plot(frames_out_ch(w).t,frames_out_ch(w).ProcessedSig)
        %     title('speech signal child');xlabel('time[s]');ylabel('amp')
        %     grid on; axis tight; hold on
        %     subplot(2,2,4)
        %     surf(frames_out_ch(w).t_for_spect,frames_out_ch(w).F,10*log10(frames_out_ch(w).PowerSpectrum),...
        %         "DisplayName","Power Spectrum","EdgeColor","none");
        %     view([0,90]); hold on
        %     axis([frames_out_ch(w).t_for_spect(1) frames_out_ch(w).t_for_spect(end) frames_out_ch(w).F(1) frames_out_Ther(w).F(end)])
        %     xlabel('Time (s)');hold on
        %     plot(frames_out_ch(w).t_for_spect,frames_out_ch(w).F1,"b","DisplayName","F_1");
        %     hold on;
        %     plot(frames_out_ch(w).t_for_spect,frames_out_ch(w).F2,"r","DisplayName","F_2");
        %     hold on;
        %     plot(frames_out_ch(w).t_for_spect,frames_out_ch(w).F3,"k","DisplayName","F_3");
        %     hold on;
        %     plot(frames_out_ch(w).t_for_spect,frames_out_ch(w).F4,"g","DisplayName","F_4");
        %     ylabel('Frequency (Hz)')
        %     c = colorbar;
        %     c.Label.String = 'Power (dB)';
        %     title("child")
        %     legend

        F1_ther = [F1_ther all(file).frames_out_Therapist(w).F1];
        F2_ther = [F2_ther all(file).frames_out_Therapist(w).F2];
        F3_ther = [F3_ther all(file).frames_out_Therapist(w).F3];
        F4_ther = [F4_ther all(file).frames_out_Therapist(w).F4];


        % soundsc(frames_out_Ther(w).ProcessedSig,Fs);
        % soundsc(frames_out_ch(w).ProcessedSig,Fs);
    end
    mF1_th(file) = mean(F1_ther(~isnan(F1_ther)));
    mF2_th(file) = mean(F2_ther(~isnan(F2_ther)));
    mF3_th(file) = mean(F3_ther(~isnan(F3_ther)));
    mF4_th(file) = mean(F4_ther(~isnan(F4_ther)));



    for w = 1:length(all(file).frames_out_child)

        F1_ch = [F1_ch all(file).frames_out_child(w).F1];
        F2_ch = [F2_ch all(file).frames_out_child(w).F2];
        F3_ch = [F3_ch all(file).frames_out_child(w).F3];
        F4_ch = [F4_ch all(file).frames_out_child(w).F4];

    end
    mF1_ch(file) = mean(F1_ch(~isnan(F1_ch)));
    mF2_ch(file) = mean(F2_ch(~isnan(F2_ch)));
    mF3_ch(file) = mean(F3_ch(~isnan(F3_ch)));
    mF4_ch(file) = mean(F4_ch(~isnan(F4_ch)));
    Process_time_one_recording = toc;
    
    myFolders = split(Autism_data(file).name,"_");
   sprintf("patient num %s from date %s is done. Process recording time is %d"...
       ,myFolders{1},myFolders{2},Process_time_one_recording)

end
all = all(3:end);
save('Recs_for_cry_scream_18092022.mat','all', '-v7.3')