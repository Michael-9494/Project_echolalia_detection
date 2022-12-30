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
from scipy import stats as st
from scipy.interpolate import interp1d
if __name__ == '__main__':
    
    
    
    # Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream
    Path =r'D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream' 
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
    time_cond = ((df[1]-df[0])<3) &((df[1]-df[0])>0.11)
    Logic_echo = ( ((df[2] == "Therapist") |  (df[2] == "Therapist2") | (df[2] == "Child") ) & 
                  (df[3] == "Echolalia") &( time_cond ))
    Logic_echo_ter = ( ((df[2] == "Therapist") |  (df[2] == "Therapist2") ) & 
                      (df[3] == "Echolalia") &( time_cond) )
    Logic_echo_child = (( (df[2] == "Child") ) &  (df[3] == "Echolalia") &( time_cond) )
    
    How_many_therapist_echo = df[ Logic_echo_ter].reset_index(drop=True)
    How_many_child_echo = df[ Logic_echo_child].reset_index(drop=True)
# drop=True
    ADOS_echo = df[Logic_echo].reset_index(drop=True)


    Logic_all = ( ( (df[2] == "Therapist") |  (df[2] == "Therapist2") |  (df[2] == "Child") ) 
                         &( time_cond ))
    Logic_all_ter = ( ((df[2] == "Therapist") |  (df[2] == "Therapist2") )  
                     &(time_cond))
    Logic_all_child = (( (df[2] == "Child") ) & 
                       ( time_cond ))
    ADOS_child = df[ Logic_all_child].reset_index(drop=True)
    ADOS_therapist = df[ Logic_all_ter].reset_index(drop=True)

    ADOS_all = df[Logic_all].reset_index(drop=True)


    
  
        # read the wav file
    sound = parselmouth.Sound(path_filename_wav) 
    Fs = sound.get_sampling_frequency()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram 
    pre_emphasized_sound = sound.copy() 
    pre_emphasized_sound.pre_emphasize() 
    time_step =0.01       #     time step(s) np.round(*Fs) 
    window_length =0.02    # window length(s) np.round(*Fs)

 
    
    out_dictionery1 = Preprocess.process_frames(ADOS_echo,pre_emphasized_sound,window_length,time_step)

   # {'F1':{},'F2':{},'F3':{},'intensity':{},
    #                   'F1_mean':{},'F2_mean':{},'F3_mean':{},
    #                   'F1_median':{},'F2_median':{},'F3_median':{},
    #                   'time_f':{},'Event':{},'Speaker':{},
    #                   'speech_data_time':{},'speech_data':{},
    #                   'pitch':{},'spectrogram':{},'pitch_obj':{}}
    F1 = []
    F2 = []
    F3 = []
    F1_new = []
    F2_new = []
    F3_new = []
    pitchh = []
    # time_f = []
    T_spect = [];
    for i in range (0,len(out_dictionery1['time_f'])):
        
        
        time_f = out_dictionery1['time_f'][i]
        
        t_new = np.linspace(time_f[0],
                            time_f[-1],
                            num=out_dictionery1['spectrogram'][i].values.shape[1],
                            endpoint=True)
        
        t_pitch = out_dictionery1['pitch_obj'][i].xs()
        pitch = out_dictionery1['pitch_obj'][i].selected_array['frequency']
        pitch_interp =interp1d(t_pitch,pitch, kind='linear')(t_new) 
        pitchh.append( pitch_interp)
        
        
        # time_f.append(out_dictionery1['time_f'][i])
        FF1 =interp1d(time_f, out_dictionery1['F1'][i], kind='linear')(t_new)
        
        FF1[pitch_interp==0] = np.nan
        F1_new.append(FF1)
        FF2 = interp1d(time_f, out_dictionery1['F2'][i],kind='linear')(t_new)
        FF2[pitch_interp==0] = np.nan
        F2_new.append(FF2)
        FF3 = interp1d(time_f, out_dictionery1['F3'][i],kind='linear')(t_new)
        FF3[pitch_interp==0] = np.nan
        F3_new.append(FF3)
        pitch_interp[pitch_interp==0] = np.NaN

        F1.append(out_dictionery1['F1'][i])
        F2.append(out_dictionery1['F2'][i])
        F3.append(out_dictionery1['F3'][i])
        tt = out_dictionery1['t_spect'][i]
        if i>0:
            t11 = tt-tt[0]+T_spect[i-1][-1]
            T_spect.append(t11[0:-1])
        else:
            t11 = tt-tt[0]
            T_spect.append(t11[0:-1])
                    
        
            
        plt.figure()
        plt.plot(out_dictionery1['speech_data_time'][i], 
                  out_dictionery1['speech_data'][i],
                  label=str(out_dictionery1['Speaker'][i]+'_'+ out_dictionery1['Event'][i]))
        plt.legend()
        plt.figure()
        Preprocess.draw_spectrogram(out_dictionery1['spectrogram'][i])
        # plt.twinx() 
        plt.plot(time_f,
                  F1[i], 'o', markersize=3,color='g', label=" F-1")
        plt.plot(time_f,
                  F2[i], 'o', markersize=3,color='m', label=" F-2")
        plt.plot(time_f,
                  F3[i], 'o', markersize=3,color='b', label=" F-3")
        plt.legend()
        plt.twinx() 
        Preprocess.draw_pitch(out_dictionery1['pitch_obj'][i]) 
        
        # plt.xlim([snd_part.xmin, snd_part.xmax]) 
        plt.title(str(out_dictionery1['Speaker'][i]+'_'+ voiceID))
        # Preprocess.draw_intensity(out_dictionery1['intensity'][i]) 
        # plt.xlim([snd.xmin, snd.xmax])
        plt.show()
        plt.figure()
        Preprocess.draw_spectrogram(out_dictionery1['spectrogram'][i])
        # plt.twinx() 
        plt.plot(t_new,
                  F1_new[i], 'o', markersize=3,color='g', label=" F-1")
        plt.plot(t_new,
                  F2_new[i], 'o', markersize=3,color='m', label=" F-2")
        plt.plot(t_new,
                  F3_new[i], 'o', markersize=3,color='b', label=" F-3")
        plt.legend()
        plt.twinx() 
        plt.plot(t_new,
                  pitch_interp, 'o', markersize=5, color='w', label=" pitch") 
        plt.plot(t_new,
                  pitch_interp, 'o', markersize=2, color='b', label=" pitch")
        # plt.xlim([snd_part.xmin, snd_part.xmax]) 
        plt.grid(False) 
        plt.ylim(0, out_dictionery1['pitch_obj'][i].ceiling) 
        plt.ylabel("fundamental frequency [Hz]")
        plt.title(str(out_dictionery1['Speaker'][i]+'_'+ voiceID))
        # Preprocess.draw_intensity(out_dictionery1['intensity'][i]) 
        # plt.xlim([snd.xmin, snd.xmax])
        plt.show()

        i+=1
        
        
    # out_dictionery2 = Preprocess.process_frames(d2,pre_emphasized_sound,window_length,time_step)
    
    # F1_all = []
    # F2_all = []
    # F3_all = []
    # F1_new_all = []
    # F2_new_all = []
    # F3_new_all = []
    # pitchh_all = []
    # # time_f = []
    # T_spect_all = [];    
    # for i in How_many_therapist['index']:
    #     print(i)
    #     time_f = out_dictionery2['time_f'][i]
        
    #     t_new = np.linspace(time_f[0],
    #                         time_f[-1],
    #                         num=out_dictionery2['spectrogram'][i].values.shape[1],
    #                         endpoint=True)
        
    #     t_pitch = out_dictionery2['pitch_obj'][i].xs()
    #     pitch = out_dictionery2['pitch_obj'][i].selected_array['frequency']
    #     pitch_interp =interp1d(t_pitch,pitch, kind='linear')(t_new) 
    #     pitchh_all.append( pitch_interp)
        
    #     # FF1 =interp1d(time_f, out_dictionery1['F1'][i], kind='linear')(t_new)        
    #     # FF1[pitch_interp==0] = np.nan
    #     # F1_new.append(FF1)
    #     # FF2 = interp1d(time_f, out_dictionery1['F2'][i],kind='linear')(t_new)
    #     # FF2[pitch_interp==0] = np.nan
    #     # F2_new.append(FF2)
    #     # FF3 = interp1d(time_f, out_dictionery1['F3'][i],kind='linear')(t_new)
    #     # FF3[pitch_interp==0] = np.nan
    #     # F3_new.append(FF3)
    #     # pitch_interp[pitch_interp==0] = np.NaN

    #     # F1.append(out_dictionery1['F1'][i])
    #     # F2.append(out_dictionery1['F2'][i])
    #     # F3.append(out_dictionery1['F3'][i])
    #     # tt = out_dictionery1['t_spect'][i]
    #     # if i>0:
    #     #     t11 = tt-tt[0]+T_spect[i-1][-1]
    #     #     T_spect.append(t11[0:-1])
    #     # else:
    #     #     t11 = tt-tt[0]
    #     #     T_spect.append(t11[0:-1])
    #     i+=1
    
    
    
    
    
    """
    move on to s-DTW based on features. I will take
    
    1). MFCC-39 (13 coef, 13 delta,13 delta-delta) (39 x time_from_spectrogram)
    2). Formants array (3 x time_from_spectrogram) 
    3). Pitch array (1 x time_from_spectrogram)
    
    """
    
    f1 = out_dictionery1['f'][1]
    
    spectrogram =  (out_dictionery1['spectrogram'][0].values)
    # filtarbank specifications:
    low_freq_mel = 0
    NFFT = 1024
    nfilt = 13  
    # create filterbank 
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
    
    
    mfcc1 = DCT_mat @ np.log10(filter_banks)
    mfcc_delta1 = scipy.signal.savgol_filter(mfcc1, window_length=9, polyorder=1, deriv=1, delta=1.0, axis=- 1)
    # mfcc_delta1 =  np.diff(mfcc, n=1,axis=1)
    mfcc_delta2 = scipy.signal.savgol_filter(mfcc_delta1, window_length=9, polyorder=1, deriv=1, delta=1.0, axis=- 1)
    mfcc_delta1_delta2 =np.concatenate((mfcc1,mfcc_delta1,mfcc_delta2),axis=0)
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row."""

    formants_f1 = F1_new[0]
    formants_f2 = F2_new[0] 
    formants_f3 = F3_new[0]
    formants =  np.array([formants_f1,formants_f2,formants_f3])
    Pitch =np.array([ pitchh[0]])
    mfcc_and_F_and_P =np.concatenate((mfcc_delta1_delta2,formants,Pitch),axis=0)
    mfcc_and_F_and_P = np.where( np.isnan(mfcc_and_F_and_P), np.finfo(float).eps, mfcc_and_F_and_P)

    # global cepstral mean and variance normalization (CMVN) on input feature vector mfcc_and_F_and_P.T  
    mfcc_and_F_and_P_Zscore  =Preprocess.CMVN(mfcc_and_F_and_P.T, variance_normalization=True)
    mfcc_and_F_and_P_Zscore = mfcc_and_F_and_P_Zscore.T
    # Normalizes the columns of a feature sequence
    X =  Preprocess.normalize_feature_sequence(mfcc_and_F_and_P_Zscore,threshold=1e-05)
    
    # X = Preprocess.quantize_matrix(x)
      
        
    # time_diff = d['End_time'][0:-1]-d['Start_time'][0:-1]
    spect_2 = out_dictionery1['spectrogram'][1].values
    new_echo_chain = np.concatenate([out_dictionery1['spectrogram'][x].values.T for x in range(0,len(out_dictionery1['spectrogram']))])
    new_echo_chain_formants_f1 = np.concatenate([F1_new[x] for x in range(0,len(F1_new))])
    new_echo_chain_formants_f2 = np.concatenate([F2_new[x] for x in range(0,len(F2_new))])
    new_echo_chain_formants_f3 = np.concatenate([F3_new[x] for x in range(0,len(F3_new))])
    new_echo_chain_formants =  np.array([new_echo_chain_formants_f1,new_echo_chain_formants_f2,new_echo_chain_formants_f3])
    new_echo_chain_Pitch =np.array([ np.concatenate([pitchh[x] for x in range(0,len(pitchh))])])
    t_new_spect = np.concatenate([T_spect[x] for x in range(0,len(T_spect))])
    t_new_spect = t_new_spect-t_new_spect[0]

    # 
    filter_banks_other = fbank @ new_echo_chain.T
    filter_banks_other = np.where(filter_banks_other == 0, np.finfo(float).eps, filter_banks_other)  # Numerical Stability
   
    mfcc_other1 = DCT_mat @ np.log10(filter_banks_other)
    mfcc_delta1_other1 = scipy.signal.savgol_filter(mfcc_other1, window_length=9, polyorder=1, deriv=1, delta=1.0, axis=- 1)
    # mfcc_delta1 =  np.diff(mfcc, n=1,axis=1)
    mfcc_delta2_other1 = scipy.signal.savgol_filter(mfcc_delta1_other1, window_length=9, polyorder=1, deriv=1, delta=1.0, axis=- 1)
    mfcc_other_delta1_delta2 =np.concatenate((mfcc_other1,mfcc_delta1_other1,mfcc_delta2_other1),axis=0)

    mfcc_other_and_F_and_P =np.concatenate((mfcc_other_delta1_delta2,new_echo_chain_formants,new_echo_chain_Pitch),axis=0)
   
    mfcc_other_and_F_and_P = np.where( np.isnan(mfcc_other_and_F_and_P), np.finfo(float).eps, mfcc_other_and_F_and_P)
    
    mfcc_other_and_F_and_P_Zscoe  =Preprocess.CMVN(mfcc_other_and_F_and_P.T, variance_normalization=True)
    mfcc_other_and_F_and_P_Zscoe = mfcc_other_and_F_and_P_Zscoe.T 
    Y =Preprocess.normalize_feature_sequence( mfcc_other_and_F_and_P_Zscoe,threshold=1e-05)
    # Y = Preprocess.quantize_matrix(y)_and_F_and_P mfcc_other_and_F_and_P.T
    
  
    C = 1 - X.T @ Y         #Cost matrix   
     
    stepsize =2  
    #   Accumulated cost matrix
    if stepsize == 1:
        D = Preprocess.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = Preprocess.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N        #  DTW-based matching function
 
    rho = N//2  #This ensures that the subsequent matches do not overlap by more than half the query length
    tau = 0.5   #Threshold for maximum Delta value allowed for matches (Default value = 0.2)
    num = 2 # number of matches
    pos = Preprocess.mininma_from_matching_function(Delta,
                                                    rho=rho,
                                                    tau=tau,
                                                    num=num)
    matches = Preprocess.matches_dtw(pos,
                                     D,
                                     stepsize=stepsize)
    
    matches_in_time =  matches/int(1/0.002)
    # T_coef=None, F_coef=None time_step*window_length
    Fs_res = Fs/NFFT
     # = np.arange(X.shape[1]) / (Fs*time_step)
     
     
     
    tempo_rel_set = [0.6,0.7, 0.8,0.9, 1.00,1.1, 1.2,1.3,1.4, 1.50]
    color_set = ['b', 'c', 'gray', 'r', 'g']
    num_tempo = len(tempo_rel_set)
    
    Delta_min, Delta_N, Delta_scale = Preprocess.matching_function_diag_multiple(X,
                                                                                  Y,
                                                                                  tempo_rel_set=tempo_rel_set,
                                                                                  cyclic=False)

    pos2 = Preprocess.mininma_from_matching_function(Delta_min,
                                                     rho=rho,
                                                     tau=tau,
                                                     num=num)
    
    matches2 = Preprocess.matches_diag(pos2, Delta_N)#//2
     
    matches2_in_time =  matches2/int(1/0.002)

     
    fig, ax = plt.subplots(2, 1,  figsize=(16, 10))
    # alpha=5, N=256, reverse=False)
    cmap = Preprocess.compressed_gray_cmap(alpha=-10,
                                           N=2*256,
                                           reverse=True)
    Preprocess.plot_matrix(C,
                           # Fs=2,
                           # Fs_F=2,
                           T_coef= t_new_spect,                           
                           F_coef=T_spect[0],
                           ax=[ax[0]],
                           ylabel='Time (seconds)',
                           title='Cost matrix $C$ with ground truth annotations (blue rectangles). step_size '+str(stepsize), 
                           colorbar=False,
                           cmap=cmap)
    
    Preprocess.plot_matches(ax[0],
                            [[0,spectrogram.shape[1]]], 
                            Delta,
                            Fr = int(1/0.002),
                            color='b',
                            s_marker='',
                            t_marker='')
    
    Preprocess.plot_matches(ax[0],
                            [[ spectrogram.shape[1] ,
                              spectrogram.shape[1]+out_dictionery1['spectrogram'][1].values.shape[1] ]], 
                            Delta,
                            Fr = int(1/0.002),
                            color='g',
                            s_marker='',
                            t_marker='')

    title = r'Matching function $\Delta_\mathrm{DTW}$ with matches (red rectangles)'
    Preprocess.plot_signal(Delta,
                           T_coef=t_new_spect ,
                           ax=ax[1],
                           color='k',
                           title=title)
    ax[1].grid()
    # ax, matches, Delta, Fs=1,
    Preprocess.plot_matches(ax[1],
                            matches,
                            Delta,
                            Fr=int(1/0.002),
                            s_marker='',
                            t_marker='o')



