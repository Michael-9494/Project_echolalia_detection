# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:47:32 2023

@author: 97254
"""


import sys
# sys.path.append(r'C:\Users\97254\anaconda3\envs\Project_speech_MP')
import numpy as np
import pandas as pd
import os
import parselmouth
from scipy.signal import resample
# import torch, torchaudio
import matplotlib.pyplot as plt
import matplotlib as mpl
# import speechaugs
import Functions


if __name__ == '__main__':

    time_step = 0.01  # time step(s) np.round(*Fs)
    window_length = 0.02    # window length(s) np.round(*Fs)
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
    fbank = Functions.compute_filterbank(
        low_freq_Mel, NFFT, nfilt, 16000)

    # Initialize lists to store the similarity scores for each event pair
    alpha_formants_list = []
    echo_list = []
    event_list = []
    Mels_list = []
    Mels_list_wa = []

    record_list = []
    spect_list = []
    spect_list_wa = []
    Mel_scores = []
    Mel_warped_scores = []
    spect_scores = []
    spect_warped_scores = []
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
    pitch_for_augment = [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4]
    # "aug_train"
    # "fold_" + str(fold)+"\\"
    save_path_aug = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\augment\\augmentations\\"
# r"D:\Recs_echolalia_26_04_2023\all"
    aug_path = r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\CODE\augment\augment_train"

    Path = r"D:\Recs_echolalia_26_04_2023\all"

    data_list = []
    #
    Database = os.listdir(Path)
    echo_num_rec = 0
    mean_seg_num = 128

    Database_aug = os.listdir(save_path_aug)
    # Filter the list to include only the pairs with the desired start

    # filtered_pairs_2 = [pair.split("_0_")  for pair in Database_aug if "20_0" in pair]

    # augment_type = list(zip(*filtered_pairs_2))[1]

    filtered_pairs = [
        pair for pair in Database_aug if "_child_pitch_shift_1" in pair]
    split_pairs = [pair.split("_child_") for pair in filtered_pairs]
    numbers = sorted_numbers = sorted(list(
        zip(*split_pairs))[0], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    # print(sorted_numbers)

    # print(numbers)

    for w in range(0, len(numbers)):
        # current_Database = Database[w]
        # print("recording num  {:.1f}".format((w +1)) + " out of {:.1f}".format(len(Database)))
        child_key = numbers[w].split("_")[0]
        child_date = numbers[w].split("_")[1]
        echo_num_rec = child_key+"_"+child_date
        # current_Database = Database[(int(echo_num_rec)-1)]

        echo_num_therapist = numbers[w].split("_")[2]
        echo_num_child = numbers[w].split("_")[2]
        print(numbers[w])
        # PitchShiftLibrosa
        for pitch_shift_therapist in pitch_for_augment:

            name_aug_ther_pitch_change = str(
                echo_num_rec) + f"_{echo_num_therapist}_terapist_pitch_shift_{'minus_' if pitch_shift_therapist < 0 else ''}{abs(pitch_shift_therapist)}.wav"

            aug_ther_path = save_path_aug + name_aug_ther_pitch_change
            try:
                augmented_terapist = parselmouth.Sound(aug_ther_path)

                (X_S_T, spect_T,
                 duration_therapist,
                 mu_p_T, mu_T) = Functions.segment_analysis(augmented_terapist,
                                                            fbank,
                                                            f0min=75,
                                                            f0max=1600,
                                                            window_length=window_length,
                                                            time_step=time_step,
                                                            nfilt=nfilt)
                X_S_T = resample(X_S_T, int(mean_seg_num), axis=1)
                spect_T = resample(spect_T, int(mean_seg_num), axis=1)

                # child

                for pitch_shift_child in pitch_for_augment:

                    name_aug_child_pitch_change = str(
                        echo_num_rec) + f"_{echo_num_child}_child_pitch_shift_{'minus_' if pitch_shift_child < 0 else ''}{abs(pitch_shift_child)}.wav"

                    aug_child_path = save_path_aug + name_aug_child_pitch_change
                    augmented_child = parselmouth.Sound(aug_child_path)

                    (Y_S_C, Mel_spectrogram_warped, spect_C, spectrogram_warped,
                     duration_child, mu_p_C, mu_C,
                     alpha_3) = Functions.segment_analysis_VTLN(augmented_child,
                                                                fbank,
                                                                mu_T,
                                                                f0min=75,
                                                                f0max=1600,
                                                                window_length=window_length,
                                                                time_step=time_step,
                                                                nfilt=nfilt, flag=0)

                    echo_list.append(1)
                    # event_list.append([event_type_T,event_type_C])
                    Y_S_C = resample(Y_S_C, int(mean_seg_num), axis=1)
                    spect_C = resample(spect_C, int(mean_seg_num), axis=1)
                    Mel_spectrogram_warped = resample(
                        Mel_spectrogram_warped, int(mean_seg_num), axis=1)
                    spectrogram_warped = resample(
                        spectrogram_warped, int(mean_seg_num), axis=1)

                    data_list.append(
                        [echo_num_rec, name_aug_ther_pitch_change, name_aug_child_pitch_change])

                    alpha = mu_T/mu_C
                    alpha_3 = alpha[2]
                    alpha_formants_list.append(alpha_3)
                    # Compute the DTW alignment and similarity score for the  features
                    (pathM, acc_cost_mat_Mel, Cost_mat_Mel, Mel_score) = Functions.DTW(X_S_T,
                                                                                       Y_S_C,
                                                                                       "euclidean")  # Y_P euclidean
                    Mel_scores.append(Mel_score)
                    # Compute the DTW alignment and similarity score for the MFCC features
                    (pathM2, acc_cost_matM2, Cost_matM2, Mel_sco2) = Functions.DTW(X_S_T,
                                                                                   Mel_spectrogram_warped,
                                                                                   "euclidean")  # Y_S_C
                    Mel_warped_scores.append(Mel_sco2)
                    # Compute the DTW alignment and similarity score for the spectrograms
                    (pathS, acc_cost_matS, Cost_matS, spect_sco) = Functions.DTW((spect_T),
                                                                                 (spect_C),
                                                                                 "euclidean")
                    spect_scores.append(spect_sco)

                    # Compute the DTW alignment and similarity score for the spectrograms
                    (pathS2, acc_cost_matS2, Cost_matS2, spect_sco2) = Functions.DTW((spect_T),
                                                                                     (spectrogram_warped),
                                                                                     "euclidean")
                    spect_warped_scores.append(spect_sco2)

            except parselmouth.PraatError as e:
                # Handle the error when reading the augmented child audio file
                print(f"Skipping due to an error: {e}")

                continue

        # echo_num_therapist +=1
        # echo_num_child +=1

    arr = (np.concatenate([(Mel_scores, Mel_warped_scores, spect_scores, spect_warped_scores, np.array(alpha_formants_list),
                            echo_list)]).T)
    save_loc = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\augment\\aug_pairs\\"

    np.save(save_loc+"record_arr_aug_all", data_list)

    df = pd.DataFrame(
        arr, columns='Mel_matchin_score Mel_warped_matchin_score spect_score_list spect_score_list_VTLN alpha_formants echo'.split())

    df.to_csv(save_loc+'DataFrame_aug.csv')

    # df = pd.DataFrame(arr,columns= 'Mel_matchin_score Mel_warped_matchin_score alpha_formants time_list echo'.split())
    # df_aug = pd.read_csv(r'C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\DataFrame_08_05_aug.csv')
    # df['echo'] = df['echo'].map({ 1: 'echolalia',0: 'no-echolalia'})
