# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:13:36 2022

@author: polonik
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:59:05 2022

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
    # for w in range(0,len(Database)): # go over the folders
    current_Database = Database[7]
    # formants_analysis_and_pitch['Folder'][w] = current_Database
    current_path = Path + "\\" + current_Database
    child_ID_and_date = os.listdir(current_path)
    
    # formants_analysis_and_pitch = {'recording_ID':{},'mean_pitch_T':{},'mean_pitch_C':{},
    #                               'mean_F_1_T':{},'mean_F_2_T':{},'mean_F_3_T':{},
    #                               'mean_F_1_C':{},'mean_F_2_C':{},'mean_F_3_C':{},
    #                               }
    # for j in range(0,len(child_ID_and_date)): # go over the current folder,'Folder':{}
    
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


        
    sound = parselmouth.Sound(path_filename_wav) 
    Fs = sound.get_sampling_frequency()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram 
    pre_emphasized_sound = sound.copy() 
    pre_emphasized_sound.pre_emphasize() 
    time_step =0.01       #     time step(s) np.round(*Fs) 
    window_length =0.02    # window length(s) np.round(*Fs)

    # ADOS_therapist.loc[i,0]
    ADOS_T_dictionery = Preprocess.process_pitch_formants(ADOS_therapist,pre_emphasized_sound,window_length,time_step)

    (mean_pitch_T,f1_mean_T,f2_mean_T,f3_mean_T,
     f1_median_T,f2_median_T,f3_median_T) = Preprocess.extract_formants_pitch(ADOS_T_dictionery)
    # Ratios of formants calculations:

    f3_F2_T = f3_mean_T/f2_mean_T    
    f2_F1_T = f2_mean_T/f1_mean_T
    f3_F1_T = f3_mean_T/f1_mean_T 
    
    
    ADOS_C_dictionery = Preprocess.process_pitch_formants(ADOS_child,pre_emphasized_sound,window_length,time_step)

    (mean_pitch_C,f1_mean_C,f2_mean_C,f3_mean_C,
     f1_median_C,f2_median_C,f3_median_C) = Preprocess.extract_formants_pitch(ADOS_C_dictionery)
    # Ratios of formants calculations:
    f3_F2_C = f3_mean_C/f2_mean_C    
    f2_F1_C = f2_mean_C/f1_mean_C
    f3_F1_C = f3_mean_C/f1_mean_C
    
    
    # alpha estimation:
#     f3_F2_T_f3_F2_C = f3_F2_T/ f3_F2_C
#     f2_F1_T_f2_F1_C = f2_F1_T/ f2_F1_C
#     f3_F1_T_f3_F1_C = f3_F1_T/ f3_F1_C
# f3_F2_T_f3_F2_C,f2_F1_T_f2_F1_C,f3_F1_T_f3_F1_C,
    
    f1_T_C = (f1_mean_T/f1_mean_C)
    f2_T_C = (f2_mean_T/f2_mean_C)
    f3_T_C = (f3_mean_T/f3_mean_C) 
    alpha_mean=(statistics.mean([f1_T_C,f2_T_C,f2_T_C])) 

    
    
    
    