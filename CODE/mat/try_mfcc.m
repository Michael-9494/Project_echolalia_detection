clear all
clc


proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
ADOS_table = readtable(proj+"\New folder\675830557_170820_new.xlsx");


fileReader = dsp.AudioFileReader(proj+"\New folder\675830557_170820.wav");
buff = dsp.AsyncBuffer;