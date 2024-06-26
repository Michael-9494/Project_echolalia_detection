
clear;close all;clc


% "D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\"
Data_Folder_out = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\";
Fs = 16000;
flag =0;
flag_sift = 0;
ads = audioDatastore(Data_Folder_out,...
  IncludeSubfolders=true,...
  FileExtensions=".wav", ...
  LabelSource="foldernames");


Autism_data = dir(Data_Folder_out); Autism_data = Autism_data(4:end);
Recs_for_cry_scream =[string(Autism_data(1).name),string( Autism_data(2).name), string(Autism_data(3).name),...
  string(Autism_data(4).name),string( Autism_data(5).name), string(Autism_data(6).name),...
  string(Autism_data(7).name), string(Autism_data(8).name),string( Autism_data(9).name)];
% Recs_for_cry_scream1 = split(Recs_for_cry_scream,"Recs_for_cry_scream");
% _25092022
Recs_for_cry_scream1 ="_25092022";

Param = struct();
% start with feature extraction
Param.alpha=exp(-(2*pi*50)/Fs); % for pre emphasis15/16
Param.WindowLength=20*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);

% all = struct();
% for i = 1:length(Recs_for_cry_scream(i))%

Data_Folder = Data_Folder_out +"Recs_for_cry_scream"+ Recs_for_cry_scream1+"\";%(i)

tic

Autism_data_in = dir(Data_Folder);
%     clear all ;
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
    %         Signal = decimate(Signal,2);
    %         Fs = 8000;
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

  all(file).frames_Therapist = splitWavByEvent(ProcessedSig, StartTherapist,EndTherapist,Fs,ADOS_table,Param);
  all(file).frames_Child = splitWavByEvent(ProcessedSig, StartChild,EndChild,Fs,ADOS_table,Param);
  ZCR_ther = [];NRG_ther = [];ZCR_ch = [];NRG_ch = [];

  for w = 1:length(all(file).frames_Therapist)
    ZCR_ther = [ZCR_ther (all(file).frames_Therapist(w).ZCR)'];
    NRG_ther = [NRG_ther (all(file).frames_Therapist(w).NRG)'];
  end
  all(file).ZCR_median_therapist = mean(ZCR_ther);
  all(file).NRG_median_therapist = mean(NRG_ther);

  for w = 1:length(all(file).frames_Child)
    ZCR_ch = [ZCR_ch (all(file).frames_Child(w).ZCR)'];
    NRG_ch = [NRG_ch (all(file).frames_Child(w).NRG)'];
  end
  all(file).ZCR_median_Child = mean(ZCR_ch);
  all(file).NRG_median_Child = mean(NRG_ch);
  %

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % finished with reading child and therapist segments
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  myFolders = split(Autism_data_in(file).name,"_");
  all(file).Child_name = myFolders{1};
  all(file).ADOS_date = myFolders{2};
  % [frames_out] = Process_frame(frames,Fs,Param,flag,VTLN,alpha)
  Process_time_one_recording = toc;
  %                 % show time elapsed
  hours = floor(Process_time_one_recording/3600);
  minuts = floor(Process_time_one_recording/60) - hours * 60;
  seconds = floor(Process_time_one_recording - hours * 3600  - minuts * 60);

  sprintf("patient num %s from date %s is done."...
    ,myFolders{1},myFolders{2})
  disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);

end
all = all(3:end);
tic
new_file = "21_12_2022_3"+Recs_for_cry_scream1+".mat";
disp(['start saving'+ string(new_file)])
save(new_file,'all', '-v7.3');%(i)

% show time elapsed
hours = floor(toc/3600);
minuts = floor(toc/60) - hours * 60;
seconds = floor(toc - hours * 3600  - minuts * 60);
disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
disp(new_file+" done and saved")

% end