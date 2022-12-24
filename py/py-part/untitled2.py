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
import sys
# sys.path.append(r'C:\Users\97254\Documents\GitHub\praat_formants_python')
import parselmouth 
from parselmouth import praat
import Preprocess
# %matplotlib inline
import statistics
from parselmouth.praat import call
from matplotlib import patches


if __name__ == '__main__':
    
    
    # D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream
    Path =r'Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream' 
    Database = os.listdir(Path)
    current_Database = Database[7]
    current_path = Path + "\\" + current_Database
    child_ID_and_date = os.listdir(current_path)
    ID = child_ID_and_date[2]
    IDc =ID.split("_")
    voiceID = IDc[0]
    
    one_child_path = current_path+ "\\" +ID
    path_excel = os.path.join('C',one_child_path+ "\\"+ID +"_new.xlsx")
    path_filename_wav = os.path.join('C',one_child_path+ "\\"+ID+".wav")
    
    df = pd.read_excel(path_excel, sheet_name='Sheet1', index_col=None, header=None)
    print(df.info())
    # Pass a list of column names

    EventStart = df[ ((df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ) & (df[3] == "Echolalia")  ][:][0]
    EventStart = EventStart.reset_index(drop=True)
    EventEnd = df[ ((df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ) & (df[3] == "Echolalia")][:][1]
    EventEnd = EventEnd.reset_index(drop=True)
    EventSpeaker = df[ ((df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child")) & (df[3] == "Echolalia") ][:][2]
    EventSpeaker = EventSpeaker.reset_index(drop=True)
    Event = df[ ((df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ) & (df[3] == "Echolalia")][:][3]
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
    Fs = sound.get_sampling_frequency()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram 
    pre_emphasized_sound = sound.copy() 
    pre_emphasized_sound.pre_emphasize() 
    
    
    d = {'Start_time': EventStart,'End_time': EventEnd,'Speaker': EventSpeaker,'Event':Event}
    d.keys()
    # {'F1':{},'F2':{},'F3':{},'intensity':{},
    #                   'F1_mean':{},'F2_mean':{},'F3_mean':{},
    #                   'F1_median':{},'F2_median':{},'F3_median':{},
    #                   'time_f':{},'Event':{},'Speaker':{},
    #                   'speech_data_time':{},'speech_data':{},
    #                   'pitch':{},'spectrogram':{},'pitch_obj':{}}
    
    time_step =0.01       #     time step(s) np.round(*Fs) 
    window_length =0.02    # window length(s) np.round(*Fs)
    out_dictionery1 = Preprocess.process_frames(d,pre_emphasized_sound,window_length,time_step)
 
 
    # for i in range (0,len(out_dictionery1['time_f'])):
        
    #     F1 = out_dictionery1['F1'][i].values
    #     F2 = out_dictionery1['F2'][i]
    #     F3 = out_dictionery1['F3'][i]
    #     plt.figure()
    #     plt.plot(out_dictionery1['speech_data_time'][i], 
    #               out_dictionery1['speech_data'][i],
    #               label=str(out_dictionery1['Speaker'][i]+'_'+ out_dictionery1['Event'][i]))
    #     plt.legend()
    #     plt.figure()
    #     Preprocess.draw_spectrogram(out_dictionery1['spectrogram'][i])
    #     # plt.twinx() 
    #     plt.plot(out_dictionery1['time_f'][i],
    #               F1, 'o', markersize=3,color='g', label=" F-1")
    #     plt.plot(out_dictionery1['time_f'][i],
    #               F2, 'o', markersize=3,color='m', label=" F-2")
    #     plt.plot(out_dictionery1['time_f'][i],
    #               F3, 'o', markersize=3,color='b', label=" F-3")
    #     plt.legend()
    #     plt.twinx() 
    #     Preprocess.draw_pitch(out_dictionery1['pitch_obj'][i]) 
    #     # plt.xlim([snd_part.xmin, snd_part.xmax]) 
    #     plt.title(str(out_dictionery1['Speaker'][i]+'_'+ voiceID))
    #     # Preprocess.draw_intensity(out_dictionery1['intensity'][i]) 
    #     # plt.xlim([snd.xmin, snd.xmax])
    #     plt.show()
        
    #     i+=1
    
    
    
    f1 = out_dictionery1['f'][1]
    tt = out_dictionery1['t_spect'][1]
    spectrogram =  (out_dictionery1['spectrogram'][0].values)
    low_freq_mel = 0
    NFFT = 1024
    nfilt = 250  
    fbank = Preprocess.compute_filterbank(low_freq_mel,NFFT,nfilt,Fs)

    filter_banks = fbank @ spectrogram
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
   
    # mel_filter_banks = 10 * np.log10(filter_banks)  # dB 
    
    # create DCT matrix
    
    (m,k) = np.mgrid[0:nfilt,0:nfilt]
    m = m+1;  # % m [1...M=nfilt]
    lamba_m = (2*m-1)/(2*nfilt)
    DCT_mat = np.sqrt(2 / nfilt) * np.cos(np.pi *lamba_m* k )
    DCT_mat[0,:] = DCT_mat[0,:] / np.sqrt(2)
    A = np.round(DCT_mat@DCT_mat.T)
    
    
    mfcc = DCT_mat @ np.log10(filter_banks)
    
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row."""
    # CMVN_mat  =Preprocess.cmvn(mfcc.T, variance_normalization=True)
    # CMVN_mat = CMVN_mat.T
    X =  Preprocess.normalize_feature_sequence(spectrogram)
    # X = Preprocess.quantize_matrix(x)
    
    
    
    first_echo_ther = out_dictionery1['speech_data'][1]
    duration1 = first_echo_ther.shape[0]/Fs
    first_echo_ther_time = out_dictionery1['speech_data_time'][1]

    
    time_diff = d['End_time'][0:-1]-d['Start_time'][0:-1]
    spect_2 = out_dictionery1['spectrogram'][1].values
    new_echo_chain = np.concatenate([out_dictionery1['spectrogram'][x].values.T for x in range(1,len(out_dictionery1['spectrogram']))])
    # new_echo_chain_time = np.concatenate([out_dictionery1['spectrogram'][x] for x in range(1,len(out_dictionery1['speech_data_time']))])
    filter_banks_other = fbank @ new_echo_chain.T
    filter_banks_other = np.where(filter_banks_other == 0, np.finfo(float).eps, filter_banks_other)  # Numerical Stability
   
    mfcc_other = DCT_mat @ np.log10(filter_banks_other)

    # CMVN_mat_other  =Preprocess.cmvn(mfcc_other.T, variance_normalization=True)
    # CMVN_mat_other = CMVN_mat_other.T 
    Y =Preprocess.normalize_feature_sequence(new_echo_chain.T)
    # Y = Preprocess.quantize_matrix(y)
    
   

    C = 1 - X.T @ Y                 #Compute cost matrix via dot product
    
    
    
    # plt.figure()
    # plt.plot(new_echo_chain_time, 
    #          new_echo_chain)
    # # label=str(out_dictionery1['Speaker'][i]+'_'+ out_dictionery1['Event'][i]))
    # plt.xlim(d['Start_time'][2],d['End_time'][2])
             
    # plt.legend()
    stepsize = 1
    
    if stepsize == 1:
        D = Preprocess.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = Preprocess.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
 
    
    pos = Preprocess.mininma_from_matching_function(Delta,
                                                    rho=2*N//3,
                                                    tau=0.4,
                                                    num=2)
    matches = Preprocess.matches_dtw(pos,
                                     D,
                                     stepsize=1)
    Fs_res = Fs/NFFT
    fig, ax = plt.subplots(2, 1,  figsize=(16, 10))
    cmap = Preprocess.compressed_gray_cmap(alpha=-10, reverse=True)
    Preprocess.plot_matrix(C, Fs=Fs_res, Fs_F=Fs_res, ax=[ax[0]], ylabel='Time (seconds)',
                         title='Cost matrix $C$ with ground truth annotations (blue rectangles)', 
                         colorbar=False,cmap=cmap)
    Preprocess.plot_matches(ax[0], [[0,570]], Delta, Fs=Fs_res,color='b')

    title = r'Matching function $\Delta_\mathrm{DTW}$ with matches (red rectangles)'
    Preprocess.plot_signal(Delta,Fs=Fs_res , ax=ax[1], color='k', title=title)
    ax[1].grid()
    # ax, matches, Delta, Fs=1,
    Preprocess.plot_matches(ax[1], matches, Delta, Fs=Fs_res, s_marker='', t_marker='o')

   
        

