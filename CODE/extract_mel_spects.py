# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:15:26 2023

@author: 97254
"""
   
import sys
sys.path.append(r'c:\program files\python39\lib\site-packages')    
import numpy as np
import pandas as pd
import os
import parselmouth 
import Functions

from scipy.signal import resample



if __name__ == '__main__':

    time_step =0.01   #     time step(s) np.round(*Fs) 
    window_length =0.02   # window length(s) np.round(*Fs)
    dynamic_range = 80
    # filtarbank specifications:
    low_freq_Mel = 0
    NFFT = 1024
    nfilt =52
    flag =0
    Fs=16000
    # create filterbank 
    fbank = Functions.compute_filterbank(low_freq_Mel,NFFT,nfilt,Fs)
            
   
    # Initialize lists to store the similarity scores for each event pair
    event_list = []
    spaker_list = []
    time_list = []
    echo_list =[]
    response_list = []
    duration_list = []
    Mels_list =[]
    record_list = []
    spect_list = []

    # Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream
    # D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_echolalia
    Path =r"D:\Recs_echolalia_26_04_2023\all" 
    Database = os.listdir(Path)
    for w in range(0,len(Database)): 
        current_Database = Database[w]
        
        # formants_analysis_and_pitch['Folder'][w] = current_Database
        current_path = Path + "\\" + current_Database
        child_ID_and_date = os.listdir(current_path)
        record_list.append(current_Database)  
       
        one_child_path = current_path+ "\\" +current_Database
        path_excel = os.path.join('C',one_child_path +"_new.xlsx")
        path_filename_wav = os.path.join('C',one_child_path+"_denoised"+".wav")
        # all_child_label = Path+ "\\" + Database[-1]
        
        
        # all_data = pd.read_excel(all_child_label, sheet_name='Eval_1', index_col=None,dtype={'Child_Key':int})+"_denoised"
        # print(all_data.info())
        
        # log =((all_data["Child_Key"]) == ( int(Database[3].split("_")[0]) ))
        # Echo_score = (all_data[log]['IM_Echo'].values)[0]
       
       
        ADOS = pd.read_excel(path_excel, sheet_name='Sheet1', index_col=None, header=None)
        print(ADOS.info())
        print("recording num  {:.1f}".format((w +1)) + " out of {:.1f}".format(len(Database)))
        # Pass a list of column names
        num_seg_max = ((3*Fs-time_step*Fs)/(window_length*Fs-time_step*Fs))
        num_seg_min = ((0.4*Fs-time_step*Fs)/(window_length*Fs-time_step*Fs))
        mean_seg_num = 128 #int(0.5*(num_seg_max+num_seg_min))
    
    
        time_cond = ((ADOS[1]-ADOS[0])<3) &((ADOS[1]-ADOS[0])>0.4)
        
        Logic_echo = ( ((ADOS[2] == "Therapist") |  (ADOS[2] == "Therapist2") | (ADOS[2] == "Child") | (ADOS[2] == "ChildEcholalia")) & 
                      (ADOS[3] == "Echolalia") &( time_cond ))
        Logic_echo_T = ( ((ADOS[2] == "Therapist") |  (ADOS[2] == "Therapist2") ) & 
                          (ADOS[3] == "Echolalia") &( time_cond) )
        Logic_echo_C = (( (ADOS[2] == "Child")  | (ADOS[2] == "ChildEcholalia")) &  (ADOS[3] == "Echolalia") &( time_cond) )
        
        ADOS_echo_T = ADOS[ Logic_echo_T].reset_index(drop=True)
        ADOS_echo_C = ADOS[ Logic_echo_C].reset_index(drop=True)
        # drop=True
        # ADOS_echo = ADOS[Logic_echo].reset_index(drop=True)
        
        
        Logic_all = ( ( (ADOS[2] == "Therapist") |  (ADOS[2] == "Therapist2") |  (ADOS[2] == "Child") ) 
                             &( time_cond ))
        Logic_all_ter = ( ((ADOS[2] == "Therapist") |  (ADOS[2] == "Therapist2") )  
                         &(time_cond))
        Logic_all_child = (( (ADOS[2] == "Child") | (ADOS[2] == "ChildEcholalia") ) & 
                           ( time_cond ))
        
        
        ADOS_child = ADOS[ Logic_all_child].reset_index(drop=True)
        ADOS_therapist = ADOS[ Logic_all_ter].reset_index(drop=True)
        
        ADOS_all = ADOS[Logic_all].reset_index(drop=True)
        
            
        # Load the audio file and extract the features
        sound = parselmouth.Sound(path_filename_wav) 
        Fs = sound.get_sampling_frequency()
        pre_emphasized_snd=sound.copy()
        pre_emphasized_snd.pre_emphasize()
    
          
        # Loop through the events in the ADOS_therapist dataframe
        for i, event_T in ADOS_therapist.iterrows():
            start_T = event_T[0]
            end_T = event_T[1]
            speaker_T = event_T[2]
            event_type_T = event_T[3]
            echo_flag = 0
            found_child_flag = 0
          # Loop through the events in the ADOS_child dataframe
            for j, event_C in ADOS_child.iterrows():
              start_C = event_C[0]    
              end_C = event_C[1]
              speaker_C = event_C[2]
              event_type_C = event_C[3]          
             
              # Check if the child event is within 5 seconds of the end of the therapist event
              if start_C >= end_T and start_C - end_T <= 10 and echo_flag == 0 and found_child_flag == 0:                    
                    
                    found_child_flag =1
                    if event_type_T=="Echolalia" and event_type_C=="Echolalia":
                        echo_list.append(1)
                        echo_flag = 1
                    else:
                        echo_list.append(0)
                    # Extract the audio segment for the event
                    response_list.append((start_C-end_T))
                    Therapist_wav_part = pre_emphasized_snd.extract_part(from_time=start_T,
                                                                 to_time =end_T,
                                                                 preserve_times=True )
                   
                    X_S_T,spect_T = Functions.segment_analysis_ver_2(Therapist_wav_part,
                                                                      fbank,
                                                                      window_length = window_length,
                                                                      time_step = time_step ,
                                                                      nfilt = nfilt,
                                                                      flag=echo_flag)
                    # X_S_T =  (Preprocess.ltw(X_S_T.T,mean_seg_num)).T
                    N = X_S_T.shape[1]
                    X_S_T = resample(X_S_T, int(mean_seg_num), axis=1)
                    spect_T = resample(spect_T, int(mean_seg_num), axis=1)
                    duration_therapist = end_T-start_T
                    
                    ''' child!!!!
                    '''
                    
                
                    # Extract the audio segment for the event
                    Chil_wav_part = pre_emphasized_snd.extract_part(from_time=start_C,
                                             to_time =end_C,
                                             preserve_times=True  )
                    duration_child = end_C-start_C
                    (Y_S_C,spect_C)= Functions.segment_analysis_ver_2(Chil_wav_part,
                                                                 fbank,
                                                                 window_length=window_length,
                                                                 time_step = time_step ,
                                                                 nfilt= nfilt,
                                                                 flag=echo_flag)
                    M = Y_S_C.shape[1]   
                    Y_S_C = resample(Y_S_C, int(mean_seg_num), axis=1)
                    spect_C = resample(spect_C, int(mean_seg_num), axis=1)
                    
                    duration_sum = duration_therapist+duration_child

                    Mels_list.append([X_S_T,Y_S_C])
                    spect_list.append([spect_T,spect_C])    
                    record_list.append(current_Database)
                    event_list.append((event_type_T,event_type_C))
                    spaker_list.append((speaker_T,speaker_C))
                    time_list.append((start_T,end_T,start_C,end_C))
                    duration_list.append((duration_therapist,duration_child))
        
    
   

    # df = pd.DataFrame(arr,columns= 'Mel_matchin_score Mel_warped_matchin_score alpha_formants time_list echo'.split())
    
    Mels_arr = np.array(Mels_list)
    np.save("D:\Project_echolalia_detection\2_Jul_2023\Mels_arr_all_2",Mels_arr)
    
    
    spect_arr = np.array(spect_list)
    np.save("D:\Project_echolalia_detection\2_Jul_2023\spect_arr_all_2",spect_arr)
    

    # record_arr = np.array(record_list)
    np.save("D:\Project_echolalia_detection\2_Jul_2023\record_arr_all_2",record_list)
    
    
    np.save("D:\Project_echolalia_detection\2_Jul_2023\echo_list_all_2",echo_list)
    
    np.save("D:\Project_echolalia_detection\2_Jul_2023\event_arr_all_2",event_list)
    