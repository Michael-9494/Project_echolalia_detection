# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:25:50 2022

@author: 97254
"""
import math
import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import sys
# sys.path.append(r'C:\Users\97254\Documents\GitHub\praat_formants_python')
import parselmouth 
from parselmouth import praat
import Preprocess
# %matplotlib inline
import statistics
from parselmouth.praat import call

if __name__ == '__main__':
    
    
    # Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_for_cry_scream_25092022
    # "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\"
    Database = os.listdir(r'Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream')
    current_Database = Database[7]
    current_path = r'Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream' + "\\" + current_Database
    child_ID_and_date = os.listdir(current_path)
    ID = child_ID_and_date[4]
    ID =ID.split("_")
    voiceID = ID[0]
    
    one_child_path = current_path+ "\\" +child_ID_and_date[4]
    path_excel = os.path.join('C',one_child_path+ "\\"+child_ID_and_date[4] +"_new.xlsx")
    path_filename_wav = os.path.join('C',one_child_path+ "\\"+child_ID_and_date[4]+".wav")
    
    df = pd.read_excel(path_excel, sheet_name='Sheet1', index_col=None, header=None)
    print(df.info())
    # Pass a list of column names

    EventStart = df[ (df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ][:][0]
    EventStart = EventStart.reset_index(drop=True)
    EventEnd = df[ (df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ][:][1]
    EventEnd = EventEnd.reset_index(drop=True)
    EventSpeaker = df[ (df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ][:][2]
    EventSpeaker = EventSpeaker.reset_index(drop=True)
    Event = df[ (df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ][:][3]
    Event = Event.reset_index(drop=True)
    
     
    
    EcholaliaEventTherapistStart = df[((df[3] == "Echolalia") & (df[2] == "Therapist")) | ((df[3] == "Echolalia") & (df[2] == "Therapist2")) ][:][0]
    # print("\n\nEcholalic Therapist events start\n\n\n", EcholaliaEventTherapistStart)

    EcholaliaEventTherapistEnd = df[((df[3] == "Echolalia") & (df[2] == "Therapist")) | ((df[3] == "Echolalia") & (df[2] == "Therapist2"))][:][1]
    # print("\n\nEcholalic Therapist events end\n\n\n", EcholaliaEventTherapistEnd)

    EcholaliaEventChildStart = df[(df[3] == "Echolalia") & (df[2] == "Child")][:][0]
    EcholaliaEventChildStart = EcholaliaEventChildStart.reset_index(drop=True)
    EcholaliaEventChildEnd = df[(df[3] == "Echolalia") & (df[2] == "Child")][:][1]
    EcholaliaEventChildEnd = EcholaliaEventChildEnd.reset_index(drop=True)
    SpeakerChild = df[(df[3] == "Echolalia") & (df[2] == "Child")][:][2]
    SpeakerChild =SpeakerChild.reset_index(drop=True)
    EventChild = df[(df[3] == "Echolalia") & (df[2] == "Child")][:][3]
    EventChild = EventChild.reset_index(drop=True)
    # print("\n\nEcholalic Child events start\n\n\n", EcholaliaEventChildStart)

    
    # print("\n\nEcholalic Child events end\n\n\n", EcholaliaEventChildEnd)


    
    # read the wav file
    sound = parselmouth.Sound(path_filename_wav) 
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram 
    pre_emphasized_sound = sound.copy() 
    pre_emphasized_sound.pre_emphasize() 
    
    
    d = {'Start_time': EcholaliaEventChildStart,'End_time': EcholaliaEventChildEnd,'Speaker': SpeakerChild,'Event':EventChild}
    d.keys()
    
    out_dictionery1 = Preprocess.process_frames(d,sound)
 
    
    for i in range (1,len(out_dictionery1['time_f'])):
    
        plt.figure()
        plt.plot(out_dictionery1['time_f'][i], out_dictionery1['F1'][i], 'o', markersize=3, label=" F-1")
        plt.plot(out_dictionery1['time_f'][i], out_dictionery1['F2'][i], 'o', markersize=3, label=" F-2")
        plt.plot(out_dictionery1['time_f'][i], out_dictionery1['F3'][i], 'o', markersize=3, label=" F-3")
        plt.legend()
        
        plt.figure()
        Preprocess.draw_spectrogram(out_dictionery1['spectrogram'][i])
        plt.twinx() 
        Preprocess.draw_intensity(out_dictionery1['intensity'][i]) 
        # plt.xlim([snd.xmin, snd.xmax])
        plt.show()
    
            
        plt.figure() 
        Preprocess.draw_pitch(out_dictionery1['pitch'][i]) 
        # plt.xlim([snd_part.xmin, snd_part.xmax]) 
        plt.show()
        i+=1
        
    
    
    
    
    
    
    
    
    
 
    
    
    
    
    # duration1 = call(sound, "Get total duration") # duration

    # (f1_list, f2_list, f3_list , f4_list,f1_mean,f2_mean,f3_mean,f4_mean,
    #  f1_median,f2_median,f3_median,f4_median,time_f)= Preprocess.measureFormants(
    #      sound, path_filename_wav, f0min,f0max)
    
    # (duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter,
    #      rapJitter, ppq5Jitter, ddpJitter, localShimmer,
    #      localdbShimmer, apq3Shimmer) = Preprocess.measurePitch(
    #          path_filename_wav, f0min, f0max, "Hertz")
             
      # Add the data to Pandas
    # df = pd.DataFrame(np.column_stack([file_list, duration_list, mean_F0_list, sd_F0_list, hnr_list, 
    #                                    localJitter_list, localabsoluteJitter_list, rapJitter_list, 
    #                                    ppq5Jitter_list, ddpJitter_list, localShimmer_list, 
    #                                    localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
    #                                    apq11Shimmer_list, ddaShimmer_list, f1_mean_list, 
    #                                    f2_mean_list, f3_mean_list, f4_mean_list, 
    #                                    f1_median_list, f2_median_list, f3_median_list, 
    #                                    f4_median_list]),
    #                                    columns=['voiceID', 'duration', 'meanF0Hz', 'stdevF0Hz', 'HNR', 
    #                                             'localJitter', 'localabsoluteJitter', 'rapJitter', 
    #                                             'ppq5Jitter', 'ddpJitter', 'localShimmer', 
    #                                             'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
    #                                             'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean', 
    #                                             'f3_mean', 'f4_mean', 'f1_median', 
    #                                             'f2_median', 'f3_median', 'f4_median'])
    
    # pcaData = Preprocess.runPCA(df) # Run jitter and shimmer PCA
    # df = pd.concat([df, pcaData], axis=1) # Add PCA data
    # # reload the data so it's all numbers
    # df.to_csv("processed_results.csv", index=False)
    # df = pd.read_csv('processed_results.csv', header=0)
    # df.sort_values('voiceID').head(20)  
        
    
 #    Fs, ADOS_recording = wavfile.read(path_filename_wav)
 #    length = ADOS_recording.shape[0] / Fs
 #    # print(f"\n\nlength = {length}[sec]")

     
    
 #    print("therapist first echolalia event \n\n", EcholaliaEventTherapistStart.values[0])
 #    short_len_therapist = int(
 #        (EcholaliaEventTherapistEnd.values[0] - EcholaliaEventTherapistStart.values[0]) * Fs + 1)
 #    time_vector_therapist_echolalic_event = np.linspace(EcholaliaEventTherapistStart.values[0],
 #                                        EcholaliaEventTherapistEnd.values[0], short_len_therapist)

 #    event_echolalia_len_child = int((EcholaliaEventChildEnd.values[0] - 
 #                                      EcholaliaEventChildStart.values[0]) * Fs + 1)
 #    time_vector_child_echolalic_event = np.linspace(EcholaliaEventChildStart.values[0],
 #                        EcholaliaEventChildEnd.values[0], event_echolalia_len_child)
    
        
 #    alpha=0.63;
 #    WindowLength_time=20*10**-3;  # 30 [mS] window
 #    Overlap=50;             #  %frame rate of 15 ms
 #    N =int( WindowLength_time*Fs); # [sec]*[sample/sec]=[sample]
 #    Hopsize =int( ((Overlap)*N)//100 )
    
    
 #    Window = np.hamming(N)
 #    ProcessedSig_ADOS,FramedSig = Preprocess.first_PreProcess(ADOS_recording,Fs,alpha,WindowLength_time,Overlap)
    
 #    therapist_part_audio = ProcessedSig_ADOS[int(EcholaliaEventTherapistStart.values[0] * Fs):
 #                    int(EcholaliaEventTherapistEnd.values[0] * Fs)+1]
        
 #    child_part_audio =ProcessedSig_ADOS[int(EcholaliaEventChildStart.values[0] * Fs):int(
 #            EcholaliaEventChildEnd.values[0] * Fs)+1]
    
    
    
 #    plt.subplot(1, 2, 1)
 #    # librosa.display.waveplot(therapist_part_audio, alpha=0.5, sr= Fs, x_axis="time",
 #    #                            offset= time_vector_therapist_echolalic_event[0])
 #    plt.plot(time_vector_therapist_echolalic_event, therapist_part_audio, label=" speech-therapist")
 #    plt.legend()
 #    plt.xlabel("Time [s]")
 #    plt.ylabel("Amplitude")
 #    plt.subplot(1, 2, 2)
 #    plt.plot(time_vector_child_echolalic_event, child_part_audio, label=" speech-child")
 #    plt.tight_layout()
 #    plt.legend()
 #    plt.xlabel("Time [s]")
 #    plt.ylabel("Amplitude")
    
 #    path = path_filename_wav
 #    newPath = path.replace(os.sep, '/')
 #    os.getcwd()
 #    # pfp.formants_at_interval(newPath, EcholaliaEventChildStart.values[1] )
    
 # # "C:\Users\97254\Desktop\Praat.exe", EcholaliaEventChildEnd.values[1]

    
    # X = Preprocess.stft(therapist_part_audio, Window, int(Hopsize), only_positive_frequencies=True)
    # Y = np.abs(X) ** 2
    # eps = np.finfo(float).eps
    # Y_db = 10 * np.log10(Y + eps)
    # # t = np.arange(short_len_therapist) / Fs
     
    # T_coef = np.arange(X.shape[1]) * Hopsize / Fs
    # F_coef = np.arange(X.shape[0]) * Fs / N
    
    
    
    
    # fig = plt.figure(figsize=(8, 5))
    
    # gs = matplotlib.gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2], width_ratios=[100, 2])
    # ax1, ax2, ax3, ax4, ax5, ax6 = [plt.subplot(gs[i]) for i in range(6)]
    
    
    # ax1.plot(time_vector_therapist_echolalic_event, therapist_part_audio, c='b')
    # ax1.set_xlim([min(time_vector_therapist_echolalic_event), max(time_vector_therapist_echolalic_event)])
    
    # ax2.set_visible(False)
    
    # left = min(T_coef)
    # right = max(T_coef) + N / Fs
    # lower = min(F_coef)
    # upper = max(F_coef)
    
    # im1 = ax3.imshow(Y, origin='lower', aspect='auto',  
    #                   extent=[left, right, lower, upper])

    # ax3.set_ylabel('Frequency (Hz)')
    # cbar = fig.colorbar(im1, cax=ax4)
    # ax4.set_ylabel('Magnitude (linear)', rotation=90)
    
    # im2 = ax5.imshow(Y_db, origin='lower', aspect='auto', 
    #                   extent=[left, right, lower, upper])

    # ax5.set_xlabel('Time (seconds)')
    # ax5.set_ylabel('Frequency (Hz)')
    # cbar = fig.colorbar(im2, cax=ax6)
    # ax6.set_ylabel('Magnitude (dB)', rotation=90)
    
    # plt.tight_layout()
              
    # # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # # D = librosa.amplitude_to_db(np.abs(librosa.stft(therapist_part_audio)), ref=np.max)
    # # img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
    # #                                sr=Fs, ax=ax[0])
    # # ax[0].set(title='Linear-frequency power spectrogram')
    # # ax[0].label_outer()
    
    # # hop_length = 1024
    # # D = librosa.amplitude_to_db(np.abs(librosa.stft(therapist_part_audio, hop_length=hop_length)),
    # #                             ref=np.max)
    # # librosa.display.specshow(D, y_axis='log', sr=Fs, hop_length=hop_length,
    # #                          x_axis='time', ax=ax[1])
    # # ax[1].set(title='Log-frequency power spectrogram')
    # # ax[1].label_outer()
    # # fig.colorbar(img, ax=ax, format="%+2.f dB")
            
    # plt.show()
    
    # # Feature extraction:
        
    # rms_therapist_part_audio = librosa.feature.rms(therapist_part_audio, frame_length=N, hop_length=Hopsize)[0]
    # frames = range(len(rms_therapist_part_audio))
    # t = librosa.frames_to_time(frames, hop_length=Hopsize)
    
    # plt.figure(figsize=(15, 17))

    # librosa.display.waveplot(therapist_part_audio, alpha=0.7, label="therapist")
    # plt.plot(t, rms_therapist_part_audio, color="r", label="rms_therapist")
    # # plt.ylim((-1, 1))
    # plt.title("Debusy")
    # plt.legend()
    
    # rms_therapist_part_audio1 = Preprocess.rmse(therapist_part_audio, frame_size=N, hop_length=Hopsize)
    # frames = range(len(rms_therapist_part_audio))
    # t = librosa.frames_to_time(frames, hop_length=Hopsize)
    
    # plt.figure(figsize=(15, 17))

    # librosa.display.waveplot(therapist_part_audio, alpha=0.7, label="therapist")
    # plt.plot(t, rms_therapist_part_audio, color="r", label="rms_therapist")
    # # plt.ylim((-1, 1))
    # plt.title("Debusy")
    # plt.legend()
    # plt.show()
    
    # zcr_therapist = librosa.feature.zero_crossing_rate(therapist_part_audio, frame_length=N, hop_length=Hopsize)[0]
    # plt.figure(figsize=(15, 10))

    # plt.plot(t, zcr_therapist, color="r")
  
    # # Preprocess.plot_spectrogram(Y_db, Fs, Hopsize)
    
    
    
    # # Ws=25*10**(-3)*Fs
    # # Ol=Ws//2;
    # # L=math.floor((len(therapist_part_audio)-Ol)/Ol);
    # # N=14;

    # # ccs=np.zeros(N,L);
    # # for n in range(L):
    # #     seg = therapist_part_audio[1+(n-1)*Ol:Ws+(n-1)*Ol]
    # #     ccs[:,n] = Preprocess.mfcc_model(seg*np.hamming(Ws),20,N,Fs)
        
