# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:33:23 2023

@author: 97254
"""

import parselmouth
from scipy.signal import resample
import Functions
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\97254\anaconda3\envs\Project_speech_MP')


if __name__ == '__main__':

    time_step = 0.01  # time step(s) np.round(*Fs)
    window_length = 0.02  # window length(s) np.round(*Fs)
    dynamic_range = 80
    # filtarbank specifications:
    low_freq_Mel = 0
    NFFT = 1024
    nfilt = 52
    f0min = 75
    f0max = 1600
    flag = 0
    Fs = 16000
    # create filterbank
    fbank = Functions.compute_filterbank(low_freq_Mel, NFFT, nfilt, 16000)

    # Initialize lists to store the similarity s cores for each event pair
    alpha_formants_list = []
    echo_list = []
    event_list = []
    Mels_list = []
    Mels_list_wa = []
    data_list = []
    record_list = []
    spect_list = []
    spect_list_wa = []

    """#     Positive Values of n_steps (Shift Up):
    # 1: minor second. # 2:  major second (whole step).
    # 3: minor third.  # 4:  major third.
    # 5: perfect fourth. # 6: tritone.
    # 7: perfect fifth. # 8: minor sixth.
    # 9: major sixth. # 10: minor seventh.
    # 11: major seventh.  # 12: octave.
    
    # Negative Values of n_steps (Shift Down):
        
    # -1:  minor second. # -2: major second (whole step).
    # -3: minor third. # -4: major third.
    # -5: perfect fourth.# -6: tritone.
    # -7: perfect fifth. # -8: minor sixth.
    # -9: major sixth. # -10: minor seventh.
    # -11: major seventh. # -12: octave."""
    pitch_for_augment = [1, 2, 3, 4, 5, 6, 7, - 1, -2, -3, -4]

    aug_path = r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\CODE\augment\augment_train"
# D:\Project_echolalia_detection\Jul_2_2023\Code\augment\augment_train
    # Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream
    # D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_echolalia
    Path = r"D:\Recs_echolalia_26_04_2023\all"
    Database = os.listdir(Path)
    echo_num_rec = 0
    mean_seg_num = 128
    for w in range(0, len(Database)):
        current_Database = Database[w]
        print(f"current_Database : {current_Database}\n")
        # formants_analysis_and_pitch['Folder'][w] = current_Database

        current_path = Path + "\\" + str(current_Database)
        child_ID_and_date = os.listdir(current_path)
        record_list.append(current_Database)

        one_child_path = current_path + "\\" + str(current_Database)
        path_excel = os.path.join('C', one_child_path + "_new.xlsx")
        path_filename_wav = os.path.join(
            'C', one_child_path+"_denoised"+".wav")

        ADOS = pd.read_excel(path_excel, sheet_name='Sheet1',
                             index_col=None, header=None)
        # print(ADOS.info())
        print("recording num  {:.1f}".format((w + 1)) +
              " out of {:.1f}".format(len(Database)))
        # Pass a list of column names

        time_cond = ((ADOS[1]-ADOS[0]) < 3) & ((ADOS[1]-ADOS[0]) > 0.4)

        Logic_all_ter = (((ADOS[2] == "Therapist") | (ADOS[2] == "Therapist2"))
                         & (time_cond))
        Logic_all_child = (((ADOS[2] == "Child") | (ADOS[2] == "ChildEcholalia")) &
                           (time_cond))

        ADOS_child = ADOS[Logic_all_child].reset_index(drop=True)
        ADOS_therapist = ADOS[Logic_all_ter].reset_index(drop=True)

        # Load the audio file and extract the features
        sound = parselmouth.Sound(path_filename_wav)
        Fs = sound.get_sampling_frequency()
        pre_emphasized_snd = sound.copy()
        pre_emphasized_snd.pre_emphasize()
        echo_num_rec += 1
        echo_num = 0

        # Loop through the events in the ADOS_therapist dataframe
        for i, event_T in ADOS_therapist.iterrows():
            start_T = event_T[0]
            end_T = event_T[1]
            speaker_T = event_T[2]
            event_type_T = event_T[3]
            echo_flag = 0
            # found_child_flag = 0
            # Loop through the events in the ADOS_child dataframe
            for j, event_C in ADOS_child.iterrows():
                start_C = event_C[0]
                end_C = event_C[1]
                speaker_C = event_C[2]
                event_type_C = event_C[3]
                # Check if the child event is within 5 seconds of the end of the therapist eventand found_child_flag == 0
                if start_C >= end_T and start_C - end_T <= 10 and echo_flag == 0:
                    # Extract the audio segment for the event
                    # found_child_flag = 1
                    if event_type_T == "Echolalia" and event_type_C == "Echolalia":
                        echo_list.append(1)
                        event_list.append([event_type_T, event_type_C])

                        Therapist_wav_part = pre_emphasized_snd.extract_part(from_time=start_T,
                                                                             to_time=end_T,
                                                                             preserve_times=False)

                        cur_path_therapist = str(
                            current_Database)+"_" + str(echo_num)+"_"+"terapist"+"_"+".wav"

                        Therapist_wav_part.save(
                            aug_path+"\\"+cur_path_therapist, 'WAV')

                        ''' child!!!!
                        '''

                        # # Extract the audio segment for the event
                        Chil_wav_part = pre_emphasized_snd.extract_part(from_time=start_C,
                                                                        to_time=end_C,
                                                                        preserve_times=False)

                        cur_path_child = str(
                            current_Database)+"_" + str(echo_num)+"_"+"child"+"_"+".wav"

                        Chil_wav_part.save(aug_path+"\\"+cur_path_child, 'WAV')

                        echo_flag = 1
                        echo_num += 1
