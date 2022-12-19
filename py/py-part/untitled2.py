# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:25:50 2022

@author: 97254
"""
import math
import numpy as np
import pandas as pd
import os
import wave
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import pyaudio
import wave
import os
import IPython.display as ipd
import librosa
import librosa.display
import sys
sys.path.append(r'C:\Users\97254\Documents\GitHub\praat_formants_python')
import praat_formants_python as pfp
sys.path.append(r"C:\Users\97254\Praat")

import Preprocess
# %matplotlib inline


if __name__ == '__main__':
    
    
    # Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_for_cry_scream_25092022
    # "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\"
    Database = os.listdir(r'Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream')
    current_Database = Database[7]
    current_path = r'Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream' + "\\" + current_Database
    current_folder = os.listdir(current_path)
    
    one_child_path = current_path+ "\\" +current_folder[1]
    path_excel = os.path.join('C',one_child_path+ "\\"+current_folder[1] +"_new.xlsx")
    path_filename_wav = os.path.join('C',one_child_path+ "\\"+current_folder[1]+".wav")
    
    df = pd.read_excel(path_excel, sheet_name='Sheet1', index_col=None, header=None)
    print(df.info())
    # Pass a list of column names
    start_column = df[0]
    end_column = df[1]
    speaker_column = df[2]
    event_column = df[3]

    # print("start_column\n", start_column)
    # print("speaker_column\n", speaker_column)
    # print("\n\n")

    EcholaliaEventTherapistStart = df[((df[3] == "Echolalia") & (df[2] == "Therapist")) | ((df[3] == "Echolalia") & (df[2] == "Therapist2")) ][:][0]
    # print("\n\nEcholalic Therapist events start\n\n\n", EcholaliaEventTherapistStart)

    EcholaliaEventTherapistEnd = df[((df[3] == "Echolalia") & (df[2] == "Therapist")) | ((df[3] == "Echolalia") & (df[2] == "Therapist2"))][:][1]
    # print("\n\nEcholalic Therapist events end\n\n\n", EcholaliaEventTherapistEnd)

    EcholaliaEventChildStart = df[(df[3] == "Echolalia") & (df[2] == "Child")][:][0]
    # print("\n\nEcholalic Child events start\n\n\n", EcholaliaEventChildStart)

    EcholaliaEventChildEnd = df[(df[3] == "Echolalia") & (df[2] == "Child")][:][1]
    # print("\n\nEcholalic Child events end\n\n\n", EcholaliaEventChildEnd)

    # read the wav file
    Fs, ADOS_recording = wavfile.read(path_filename_wav)
    length = ADOS_recording.shape[0] / Fs
    # print(f"\n\nlength = {length}[sec]")

     
    
    print("therapist first echolalia event \n\n", EcholaliaEventTherapistStart.values[0])
    short_len_therapist = int(
        (EcholaliaEventTherapistEnd.values[0] - EcholaliaEventTherapistStart.values[0]) * Fs + 1)
    time_vector_therapist_echolalic_event = np.linspace(EcholaliaEventTherapistStart.values[0],
                                        EcholaliaEventTherapistEnd.values[0], short_len_therapist)

    event_echolalia_len_child = int((EcholaliaEventChildEnd.values[0] - 
                                      EcholaliaEventChildStart.values[0]) * Fs + 1)
    time_vector_child_echolalic_event = np.linspace(EcholaliaEventChildStart.values[0],
                        EcholaliaEventChildEnd.values[0], event_echolalia_len_child)
    
        
    alpha=0.63;
    WindowLength_time=20*10**-3;  # 30 [mS] window
    Overlap=50;             #  %frame rate of 15 ms
    N =int( WindowLength_time*Fs); # [sec]*[sample/sec]=[sample]
    Hopsize =int( ((Overlap)*N)//100 )
    
    
    Window = np.hamming(N)
    ProcessedSig_ADOS,FramedSig = Preprocess.first_PreProcess(ADOS_recording,Fs,alpha,WindowLength_time,Overlap)
    
    therapist_part_audio = ProcessedSig_ADOS[int(EcholaliaEventTherapistStart.values[0] * Fs):
                    int(EcholaliaEventTherapistEnd.values[0] * Fs)+1]
        
    child_part_audio =ProcessedSig_ADOS[int(EcholaliaEventChildStart.values[0] * Fs):int(
            EcholaliaEventChildEnd.values[0] * Fs)+1]
    
    
    
    plt.subplot(1, 2, 1)
    # librosa.display.waveplot(therapist_part_audio, alpha=0.5, sr= Fs, x_axis="time",
    #                            offset= time_vector_therapist_echolalic_event[0])
    plt.plot(time_vector_therapist_echolalic_event, therapist_part_audio, label=" speech-therapist")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.subplot(1, 2, 2)
    plt.plot(time_vector_child_echolalic_event, child_part_audio, label=" speech-child")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    path = path_filename_wav
    newPath = path.replace(os.sep, '/')
    os.getcwd()
    pfp.formants_at_interval(newPath, EcholaliaEventChildStart.values[1] )
    
 # "C:\Users\97254\Desktop\Praat.exe", EcholaliaEventChildEnd.values[1]

    
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
        
