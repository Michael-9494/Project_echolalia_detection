# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:47:32 2023

@author: 97254
"""

import torchaudio
import torch
import parselmouth
import librosa
import speechaugs
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import resample
import Functions
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\97254\anaconda3\envs\Project_speech_MP')


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
    fbank = Functions.compute_filterbank(low_freq_Mel, NFFT, nfilt, Fs)
    VTLN = "no_VTLN"  #
    # Initialize lists to store the similarity scores for each event pair
    alpha_formants_list = []
    echo_list = []
    event_list = []
    Mels_list = []
    Mels_list_wa = []
    record_list = []
    spect_list = []
    spect_list_wa = []
    pitch_for_augment = [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4]
    # "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\augment\\aug_train\\"
    # "fold_" + str(fold)+"\\"
    save_path_aug = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\augment\\augmentations\\"
# r"D:\Recs_echolalia_26_04_2023\all"
    aug_path = r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\CODE\augment\augment_train"

    Path = r"D:\Recs_echolalia_26_04_2023\all"
    Database = os.listdir(Path)
    echo_num_rec = 0
    mean_seg_num = 128

    Database_aug = os.listdir(save_path_aug)
    # Filter the list to include only the pairs with the desired start

    filtered_pairs = [
        pair for pair in Database_aug if "_child_pitch_shift_1" in pair]
    split_pairs = [pair.split("_child") for pair in filtered_pairs]
    numbers = sorted_numbers = sorted(list(
        zip(*split_pairs))[0],
        key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
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
                    # spect_C = resample(spect_C, int(mean_seg_num), axis=1)
                    Mel_spectrogram_warped = resample(
                        Mel_spectrogram_warped, int(mean_seg_num), axis=1)
                    # spectrogram_warped = resample(spectrogram_warped, int(mean_seg_num), axis=1)

                    # fig = plt.figure(figsize=(18, 16))
                    # gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
                    # ax1 = fig.add_subplot(gs[0])
                    # ax2 = fig.add_subplot(gs[1])
                    # ax1.set_title(name_aug_ther_pitch_change, fontsize=20)
                    # ax2.set_title(name_aug_child_pitch_change, fontsize=20)
                    # X_S_T_im = ax1.imshow(X_S_T, origin="lower", interpolation="nearest")
                    # Y_S_C_im = ax2.imshow(Y_S_C, origin="lower", interpolation="nearest")
                    # plt.show()

                    Mels_list.append([X_S_T, Y_S_C])
                    # spect_list.append([spect_T, spect_C])
                    Mels_list_wa.append([X_S_T, Mel_spectrogram_warped])
                    # spect_list_wa.append([spect_T,spectrogram_warped])
                    record_list.append(echo_num_rec)

            except parselmouth.PraatError as e:
                # Handle the error when reading the augmented child audio file
                print(f"Skipping due to an error: {e}")

                continue

    save_loc = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\augment\\aug_pairs\\"
    Mels_arr = np.array(Mels_list)
    np.save(save_loc+"Mels_arr_all_aug_no_VTLN", Mels_arr)

    Mels_arr_wa = np.array(Mels_list_wa)
    np.save(save_loc+"Mels_arr_all_aug_VTLN", Mels_arr_wa)

    # spect_arr = np.array(spect_list)
    # np.save(save_loc+"spect_arr_all_aug_72_200_no_VTLN", spect_arr)

    # spect_arr_wa =np.array(spect_list_wa)
    # np.save(save_loc+"spect_arr_all_aug_wa",spect_arr_wa)

    # record_arr = np.array(record_list)
    np.save(save_loc+"record_arr_aug_all", record_list)

    np.save(save_loc+"echo_list_aug_all", echo_list)

    # np.save(save_loc+"event_arr_aug_all_72_200_no_VTLN",event_list)
