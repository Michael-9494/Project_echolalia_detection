# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:26:33 2023

@author: 97254
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:32:12 2023

@author: 97254
"""


import parselmouth
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from scipy.signal import resample
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
from matplotlib.patches import ConnectionPatch
import Functions
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(
    r'C:/Users/97254/Documents/GitHub/Project-speech/Project_echolalia_detection/py/py-part')


if __name__ == '__main__':

    time_step = 0.01  # time step(s) np.round(*Fs)
    window_length = 0.02  # window length(s) np.round(*Fs)
    dynamic_range = 50
    # filtarbank specifications:
    low_freq_mel = 0
    NFFT = 512*2
    nfilt = 52
    f0min = 75
    f0max = 1600
    flag = 1
    Fs = 16000
    # create filterbank
    fbank = Functions.compute_filterbank(low_freq_mel, NFFT, nfilt, 16000)

    # Compute the frequency values for the x-axis
    freqs = np.linspace(0, Fs / 2, len(fbank[0]))

    # Plot the filterbank
    plt.figure(figsize=(20, 10))
    plt.plot(freqs, fbank.T)
    plt.ylabel("Weights", fontsize=28)
    plt.xlabel("Frequency [Hz]", fontsize=28)
    plt.title("Mel Filterbank", fontsize=28)
    plt.yticks(fontsize=28)
    # Set the x-axis tick labels
    # Adjust the tick frequency as needed
    plt.xticks(np.arange(0, (Fs / 2) + 1, 1000), fontsize=28)

    # Add mel-frequency labels
    # for i in range(fbank.shape[0]):
    #     for j in range(fbank.shape[1]):
    #         if fbank[i, j] == 1:
    #             mel_freq = hz_to_mel(freqs[j])
    #             plt.text(freqs[j], fbank[i, j], f'{mel_freq:.1f} ', ha='center', va='bottom')

    plt.show()

    # create DCT matrix
    (m, k) = np.mgrid[0:nfilt, 0:nfilt]
    m = m+1  # % m [1...M=nfilt]128
    lamba_m = (2*m-1)/(2*nfilt)
    DCT_mat = np.sqrt(2 / nfilt) * np.cos(np.pi * lamba_m * k)
    DCT_mat[0, :] = DCT_mat[0, :] / np.sqrt(2)
    A = np.round(DCT_mat@DCT_mat.T)

    # Define the mel scale function
    # Initialize lists to store the similarity scores for each event pair
    pitch_scores = []
    mel_scores = []
    spect_scores = []
    spect_warped_scores = []

    formants_scores = []
    alpha_formants_list = []
    mel_warped_scores = []
    time_list = []
    echo_list = []
    alpha_formants_list = []
    echo_list = []
    segmants_list = []
    Mels_list = []
    spect_list = []

    num_seg_max = ((3*Fs-time_step*Fs)/(window_length*Fs-time_step*Fs))
    num_seg_min = ((0.4*Fs-time_step*Fs)/(window_length*Fs-time_step*Fs))
    # int(0.5*(num_seg_max+num_seg_min)+1)  #num_seg_min 200#
    mean_seg_num = 128

    path_filename_wav_miss = r"D:\Recs_echolalia_26_04_2023\all\1020840187_280620\1020840187_280620.wav"

    sound_me = parselmouth.Sound(path_filename_wav_miss)
    pre_emphasized_snd = sound_me.copy()
    pre_emphasized_snd.pre_emphasize()
    segment_sound = pre_emphasized_snd.extract_part(
        from_time=199.95, to_time=211.01, preserve_times=True)
    plt.figure(figsize=(30, 15))
    plt.plot(segment_sound.xs(), segment_sound.values.T, linestyle='-')
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Amplitude [arb. unit]", fontsize=20)
    plt.title("ADOS recordong", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    segments = [
        # (196.24, 196.46, "Noise", ""),
        (196.83, 197.56, "Therapist", ""),
        # (197.65, 197.96, "Noise", ""),
        (197.96, 198.31, "Child", "Speech"),
        # (198.31, 198.91, "Therapist", ""),
        # (199.35, 199.57, "Therapist", ""),
        # (199.57, 199.95, "Simultaneous", ""),
        (199.95, 200.34, "Therapist", ""),
        (201.5, 202.24, "Simultaneous", ""),
        # (202.24, 202.96, "Noise", ""),
        # (202.96, 203.75, "Child", "Speech"),
        (205.23, 206.36, "Therapist", "Echolalia"),
        (207.11, 208.64, "Child", "Echolalia"),
        (208.92, 211.01, "Child", "Speech")
    ]

    # Set up different colors/styles for speakers and speech types
    colors_speakers = {'Simultaneous': 'm', 'Noise': 'red',
                       'Child': 'blue', 'Therapist':  'green'}
    labels_speech_types = ['Speech', 'Echolalia']

    # Plotting the pre-emphasized sound waveform
    plt.figure(figsize=(30, 15))

    # Plotting each segment with different color/style and creating handles and labels for the legend
    handles = []
    labels = []
    for segment in segments:
        from_time, to_time, speaker, speech_type = segment
        segment_sound = pre_emphasized_snd.extract_part(
            from_time=from_time, to_time=to_time, preserve_times=True)
        line, = plt.plot(segment_sound.xs(), segment_sound.values.T,
                         color=colors_speakers[speaker], linestyle='-')
        handles.append(line)
        labels.append(speaker)

        plt.text(from_time+0.5, 0.04, (f"{speaker}\n" + f" {speech_type}"),
                 color='black', fontsize=28, ha='center', va='center')

    plt.xlabel("Time [s]", fontsize=28)
    plt.ylabel("Amplitude [arb. unit]", fontsize=28)
    plt.title("ADOS recordong", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # Adding legend
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    plt.legend(unique_handles, unique_labels, loc='upper right', fontsize=20)

    plt.show()

    # r"D:\Recs_echolalia_26_04_2023\augment\train\1020840187_280620_0_terapist_echo_num.wav"
    # r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\Code\augment\aug_train\1020840187_280620_0_terapist_.wav"
    path_filename_wav_miss = r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\CODE\augment\rotza_of_miss.wav"
    sound = parselmouth.Sound(path_filename_wav_miss)
    # resample(self: parselmouth.Sound, new_frequency: float, precision: int = 50)
    Sound = sound.resample(new_frequency=16000, precision=50)

    Fs = Sound.get_sampling_frequency()

    X_S_T, spect_T, duration_therapist, mu_p_T, mu_T = Functions.segment_analysis(Sound,
                                                                                  fbank,
                                                                                  f0min=75,
                                                                                  f0max=1600,
                                                                                  window_length=window_length,
                                                                                  time_step=time_step,
                                                                                  nfilt=nfilt, flag=0)
    N = X_S_T.shape[1]
    X_S_T = resample(X_S_T, int(mean_seg_num), axis=1)
    spect_T = resample(spect_T, int(mean_seg_num), axis=1)

    ''' me!!!!
    '''

    # path_filename_wav_me =r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\augment\aug_train\1020840187_280620_0_child_echo_num_pitch_shift_minus_4.wav"
    # r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\Code\augment\aug_train\1020840187_280620_0_child_.wav"
    path_filename_wav_me = r"C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part\CODE\augment\rotza_of_me.wav"
    # Load the audio file and extract the pitch and Mel featuresor index_mee== match_4_2

    # Load the audio file and extract the pitch and Mel features
    sound_me = parselmouth.Sound(path_filename_wav_me)
    # resample(self: parselmouth.Sound, new_frequency: float, precision: int = 50)
    Sound_me = sound_me.resample(new_frequency=16000, precision=50)
    (Y_S_C, mel_spectrogram_warped, spect_C, spectrogram_warped,
     duration_child, mu_p_C, mu_C,
     alpha_3) = Functions.segment_analysis_VTLN(Sound_me,
                                                fbank,
                                                mu_T,
                                                f0min=75,
                                                f0max=1600,
                                                window_length=window_length,
                                                time_step=time_step,
                                                nfilt=nfilt, flag=0)

    M = Y_S_C.shape[1]
    Y_S_C = resample(Y_S_C, int(mean_seg_num), axis=1)
    spect_C = resample(spect_C, int(mean_seg_num), axis=1)
    spectrogram_warped = resample(
        spectrogram_warped, int(mean_seg_num), axis=1)
    mel_spectrogram_warped = resample(
        mel_spectrogram_warped, int(mean_seg_num), axis=1)

    # Compute the DTW alignment and similarity score for the Mel features
    (pathM, acc_cost_matM, Cost_matM, mel_sco) = Functions.dynamic_time_warping(X_S_T,
                                                                                Y_S_C,
                                                                                "euclidean")  # cosine euclidean

    # Compute the DTW alignment and similarity score for the Mel features
    (pathM2, acc_cost_matM2, Cost_matM2, mel_sco2) = Functions.dynamic_time_warping(X_S_T,
                                                                                    mel_spectrogram_warped,
                                                                                    "euclidean")  # Y_S_C

    # Compute the DTW alignment and similarity score for the spectrograms
    (pathS, acc_cost_matS, Cost_matS, spect_sco) = Functions.dynamic_time_warping((spect_T),
                                                                                  (spect_C),
                                                                                  "euclidean")

    # Compute the DTW alignment and similarity score for the spectrograms
    (pathS2, acc_cost_matS2, Cost_matS2, spect_sco2) = Functions.dynamic_time_warping((spect_T),
                                                                                      (spectrogram_warped),
                                                                                      "euclidean")

    # X_S_T1 =  (Preprocess.ltw(X_S_T.T, Y_S_C.shape[1])).T
    plt.figure()
    plt.plot(sound_me.xs(), sound_me.values.T)
    plt.xlim([sound_me.xmin, sound_me.xmax])
    plt.xlabel("time[s]")
    plt.ylabel("amplitude")
    plt.show()
    #
    pre_emphasized_snd = sound_me.copy()
    pre_emphasized_snd.pre_emphasize()
    plt.figure()
    plt.plot(pre_emphasized_snd.xs(), pre_emphasized_snd.values.T)
    plt.xlim([pre_emphasized_snd.xmin, pre_emphasized_snd.xmax])
    plt.xlabel("time[s]")
    plt.ylabel("amplitude")
    plt.show()

    # spectrogram=pre_emphasized_snd.to_spectrogram(window_length=0.02,
    #                                               maximum_frequency=8000)
    # # plt.figure()
    # draw_spectrogram(spectrogram)
    # plt.xlim([sound_me.xmin,sound_me.xmax])
    # plt.show()

    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title(" Matching score:{:.2f}".format(mel_sco), fontsize=20)

    # Convert y-axis to Mel scale
    fmin = 100
    fmax = 6000
    n_mels = Y_S_C.shape[0]
    mel_min = 2595 * np.log10(1 + (fmin / 700))
    mel_max = 2595 * np.log10(1 + (fmax / 700))

    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 0.6, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[1])
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax3.axis('off')

    X_S_T_im = ax1.imshow(X_S_T, origin="lower", interpolation="nearest")
    Y_S_C_im = ax2.imshow(Y_S_C, origin="lower", interpolation="nearest")
    # fig.colorbar(X_S_T_im,label=" [dB]")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(X_S_T_im, cax=cax, label=" [dB]")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05,)
    fig.colorbar(Y_S_C_im, cax=cax, label=" [dB]")

    # # Add x and y axis labels to both subplots
    ax1.set_ylabel('Frequency [Hz]', fontsize=20)
    ax1.set_yticks(np.linspace(0, n_mels, 5))
    ax1.set_yticklabels(np.round(np.linspace(fmin, fmax, 5)).astype(int),
                        fontsize=20)
    ax2.set_xlabel('Frames', fontsize=20)
    ax2.set_ylabel('Frequency [Hz]', fontsize=20)
    ax2.set_yticks(np.linspace(0, n_mels, 5))
    ax2.set_yticklabels(np.round(np.linspace(fmin, fmax, 5)).astype(int),
                        fontsize=20)

    # for x_i, y_j in pathM:
    # # Shorten the connection patch by half
    #     con = ConnectionPatch(xyA=(x_i, 0),
    #                       xyB=(y_j, Y_S_C.shape[0] - 1),
    #                       coordsA="data", coordsB="data",
    #                       axesA=ax1, axesB=ax2, color="C7")
    #     ax2.add_artist(con)
    plt.show()

    fig = plt.figure(figsize=(18, 16))
    plt.axis('off')
    plt.title(" Matching score:{:.2f}".format(mel_sco2), fontsize=20)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 0.6, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[1])
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax3.axis('off')

    X_S_T_im = ax1.imshow(X_S_T, origin="lower", interpolation="nearest")
    mel_spectrogram_warped_im = ax2.imshow(
        mel_spectrogram_warped, origin="lower", interpolation="nearest")
    # fig.colorbar(X_S_T_im,label=" [dB]")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(X_S_T_im, cax=cax, label=" [dB]")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(mel_spectrogram_warped_im, cax=cax, label=" [dB]")

    # # Add x and y axis labels to both subplots
    ax1.set_ylabel('Frequency [Hz]', fontsize=20)
    ax1.set_yticks(np.linspace(0, n_mels, 5))
    ax1.set_yticklabels(np.round(np.linspace(fmin, fmax, 5)).astype(int),
                        fontsize=16)
    ax2.set_xlabel('Frames', fontsize=20)
    ax2.set_ylabel('Frequency [Hz]', fontsize=20)
    ax2.set_yticks(np.linspace(0, n_mels, 5))
    ax2.set_yticklabels(np.round(np.linspace(fmin, fmax, 5)).astype(int),
                        fontsize=16)

    for x_i, y_j in pathM2:
        con = ConnectionPatch(xyA=(x_i, 0),
                              xyB=(y_j, mel_spectrogram_warped.shape[0] - 1),
                              coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color="C7")
        ax2.add_artist(con)

    plt.show()
