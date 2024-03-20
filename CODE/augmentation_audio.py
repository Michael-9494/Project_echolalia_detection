# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:17:31 2023

@author: 97254
"""

import torchaudio
import torch
import parselmouth
import librosa
import matplotlib as mpl
import Functions
import os
import pandas as pd
import numpy as np
import sys
# sys.path.append(r'C:\Users\97254\anaconda3\envs\Project_speech_MP')


if __name__ == '__main__':

    time_step = 0.01  # time step(s) np.round(*Fs)
    window_length = 0.02   # window length(s) np.round(*Fs)
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

    # Initialize lists to store the similarity scores for each event pair
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

    # "fold_" + str(fold)+"\\"
    save_path_aug = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\augment\\augmentations\\"
    aug_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\augment\\augment_train\\"

    # D:\Autism\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Rec s_echolalia
    Path = r"D:\Recs_echolalia_26_04_2023\all"
    Database = os.listdir(Path)
    echo_num_rec = 0
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
                # Check if the child event is within 5 seconds of the end of the therapist event
                if start_C >= end_T and start_C - end_T <= 10 and echo_flag == 0:
                    # Extract the audio segment for the event

                    if event_type_T == "Echolalia" and event_type_C == "Echolalia":
                        echo_list.append(1)
                        event_list.append([event_type_T, event_type_C])

                        cur_path_therapist = str(
                            current_Database)+"_" + str(echo_num)+"_"+"terapist_"+".wav"

                        therapist_to_augment, Fs = torchaudio.load(
                            (aug_path+cur_path_therapist))
                        # plt.plot(np.arange(0,therapist_to_augment.shape[1])/Fs, therapist_to_augment[0])
                        # plt.show()

                        # # # PitchShiftLibrosa
                        for pitch_shift in pitch_for_augment:
                            augmented = librosa.effects.pitch_shift(y=therapist_to_augment[0].numpy(),
                                                                    sr=Fs,
                                                                    n_steps=pitch_shift,
                                                                    bins_per_octave=36)

                            augmented = torch.tensor(
                                augmented, dtype=torch.float).unsqueeze(0)
                            name_aug_ther_pitch_change = str(
                                current_Database) + f"_{echo_num}_terapist_pitch_shift_{'minus_' if pitch_shift < 0 else ''}{abs(pitch_shift)}.wav"
                            torchaudio.save(
                                save_path_aug+"\\" + name_aug_ther_pitch_change, augmented, Fs)
                            augmented_terapist = parselmouth.Sound(
                                save_path_aug + name_aug_ther_pitch_change)

                        ''' child!!!!
                        '''
                        cur_path_child = str(
                            current_Database)+"_" + str(echo_num)+"_"+"child_"+".wav"
                        child_to_augment, Fs = torchaudio.load(
                            (aug_path+cur_path_child))

                        for pitch_shift in pitch_for_augment:
                            augmented = librosa.effects.pitch_shift(y=child_to_augment[0].numpy(),
                                                                    sr=Fs,
                                                                    n_steps=pitch_shift,
                                                                    bins_per_octave=36)
                            augmented = torch.tensor(
                                augmented, dtype=torch.float).unsqueeze(0)
                            name_aug_child_pitch_change = str(
                                current_Database) + f"_{echo_num}_child_pitch_shift_{'minus_' if pitch_shift < 0 else ''}{abs(pitch_shift)}.wav"
                            torchaudio.save(
                                save_path_aug + "\\" + name_aug_child_pitch_change, augmented, Fs)

                        echo_flag = 1
                        echo_num += 1
