clear;close all;clc
% "D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\";"Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\";
Data_Folder_out = "Y:\Database\Recordings_for_Segmentor\Naomi\Recordings_for_speech_enhancement";
flag =0;
flag_sift = 0;

Autism_data = dir(Data_Folder_out); Autism_file = Autism_data(4).name;

[Signal,Fs]=audioread("666885463_121020.wav");

if abs(max(max(Signal))) > 1 % check clipping
    error('Cliping detected while mixing.\n');
end


Y = whitening(Signal, 2);











