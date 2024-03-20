# -*- coding: utf-8 -*

import parselmouth
from scipy.signal import resample
import Functions
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':

    H = 0.01  # time step(s) np.round(*Fs)
    N = 0.02    # window length(s) np.round(*Fs)
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
    num_seg_max = ((3*Fs-H*Fs)/(N*Fs-H*Fs))
    num_seg_min = ((0.4*Fs-H*Fs)/(N*Fs-H*Fs))
    mean_seg_num = 128  # int(0.5*(num_seg_max+num_seg_min))
    # 128  #
    # secd = (mean_seg_num*((N-H)*Fs))/Fs + H

    secd1 = (mean_seg_num * (H)*Fs) / Fs

    # print('Time range (seconds): [%5.2f, %5.2f]; frequency range (Hertz): [%5.2f, %5.2f]'% (left, right, lower, upper))

    # Initialize lists to store the similarity scores for each event pair
    spect_scores = []
    Mel_scores = []
    Mel_warped_scores = []
    event_list = []
    spaker_list = []
    time_list = []
    echo_list = []
    all1 = []
    response_list = []
    duration_list = []
    alpha_formants_list = []
    alpha_pitch_list = []
    spect_warped_scores = []
    Mels_list = []
    record_list = []
    spect_list = []
    spect_score_list = []
    spect_score_list_VTLN = []

    Path = 'D:\\Recs_echolalia_26_04_2023\\all'
    Database = os.listdir(Path)
    all_child_label = "D:\\Recs_echolalia_26_04_2023\\All_Data_26_04_2023.xlsx"

    for w in range(0, len(Database)):

        current_Database = Database[w]
        print(f"current_Database : {current_Database}\n")

        current_path = Path + "\\" + str(current_Database)
        child_ID_and_date = os.listdir(current_path)

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
        Logic_all_child = (((ADOS[2] == "Child") |
                            (ADOS[2] == "ChildEcholalia")) & (time_cond))

        ADOS_child = ADOS[Logic_all_child].reset_index(drop=True)
        ADOS_therapist = ADOS[Logic_all_ter].reset_index(drop=True)

        # Load the audio file and extract the features
        sound = parselmouth.Sound(path_filename_wav)
        Fs = sound.get_sampling_frequency()

        # Loop through the events in the ADOS_therapist dataframe
        for l, event_T in ADOS_therapist.iterrows():
            start_T = event_T[0]
            end_T = event_T[1]
            speaker_T = event_T[2]
            event_type_T = event_T[3]
            echo_flag = 0

            # Loop through the events in the ADOS_child dataframe
            for j, event_C in ADOS_child.iterrows():
                start_C = event_C[0]
                end_C = event_C[1]
                speaker_C = event_C[2]
                event_type_C = event_C[3]

                # Check if the child event is within 10 seconds
                # of the end of the therapist event
                if start_C >= end_T and start_C - end_T <= 10 and echo_flag == 0:
                    # Extract the audio segment for the event

                    response_list.append((start_C-end_T))
                    Therapist_wav_part = sound.extract_part(from_time=start_T,
                                                            to_time=end_T,
                                                            preserve_times=True)

                    (X_S_T, spect_T,
                     duration_therapist,
                     mu_p_T, mu_T) = Functions.segment_analysis(Therapist_wav_part,
                                                                fbank,
                                                                f0min=75,
                                                                f0max=1600,
                                                                window_length=N,
                                                                time_step=H,
                                                                nfilt=nfilt)

                    # X_S_T =  (Preprocess.ltw(X_S_T.T,mean_seg_num)).T

                    X_S_T = resample(X_S_T, int(mean_seg_num), axis=1)
                    spect_T = resample(spect_T, int(mean_seg_num), axis=1)
                    duration_therapist = end_T-start_T

                    ''' child!!!!
                    '''

                    # Extract the audio segment for the event
                    Chil_wav_part = sound.extract_part(from_time=start_C,
                                                       to_time=end_C,
                                                       preserve_times=True)
                    duration_child = end_C-start_C
                    (Y_S_C, Mel_VTLN, spect_C, spectrogram_VTLN,
                     duration_child, mu_p_C, mu_C,
                     alpha_3) = Functions.segment_analysis_VTLN(Chil_wav_part,
                                                                fbank,
                                                                mu_T,
                                                                f0min=75,
                                                                f0max=1600,
                                                                window_length=N,
                                                                time_step=H,
                                                                nfilt=nfilt,
                                                                flag=0)

                    Y_S_C = resample(Y_S_C, int(mean_seg_num), axis=1)
                    spect_C = resample(spect_C, int(mean_seg_num), axis=1)
                    Mel_VTLN = resample(
                        Mel_VTLN, int(mean_seg_num), axis=1)
                    spectrogram_VTLN = resample(
                        spectrogram_VTLN, int(mean_seg_num), axis=1)

                    duration_sum = duration_therapist+duration_child

                    alphap = mu_p_T / mu_p_C
                    alpha = mu_T/mu_C
                    alpha_3 = alpha[2]
                    alpha_formants_list.append(alpha_3)

                    # Compute the DTW alignment and similarity score for the spectrograms

                    (pathM, acc_cost_mat_Mel,
                     Cost_mat_Mel, Mel_score) = Functions.DTW(X_S_T,
                                                              Y_S_C,
                                                              "euclidean")
                    Mel_scores.append(Mel_score)

                    (pathM2, acc_cost_matM2,
                     Cost_matM2, Mel_sco2) = Functions.DTW(X_S_T,
                                                           Mel_VTLN,
                                                           "euclidean")
                    Mel_warped_scores.append(Mel_sco2)

                    (pathS, acc_cost_matS,
                     Cost_matS, spect_sco) = Functions.DTW((spect_T),
                                                           (spect_C),
                                                           "euclidean")
                    spect_scores.append(spect_sco)

                    (pathS2, acc_cost_matS2,
                     Cost_matS2, spect_sco2) = Functions.DTW((spect_T),
                                                             (spectrogram_VTLN),
                                                             "euclidean")
                    spect_warped_scores.append(spect_sco2)

                    if (event_type_T == "Echolalia" and
                            event_type_C == "Echolalia"):
                        echo_list.append(1)
                        echo_flag = 1

                    else:
                        echo_list.append(0)

                    record_list.append(current_Database)
                    event_list.append((event_type_T, event_type_C))
                    spaker_list.append((speaker_T, speaker_C))
                    time_list.append((start_T, end_T, start_C, end_C))
                    duration_list.append((duration_therapist, duration_child))

    # concc_all = np.concatenate([all1[x] for x in range(0,len(all1))],axis=1)
    # # Average the similarity scores for all event pairs

    avg_Mel_score = np.mean(Mel_scores)

    start_t = np.array(time_list)[:, 0]
    end_t = np.array(time_list)[:, 1]
    start_c = np.array(time_list)[:, 2]
    end_c = np.array(time_list)[:, 3]

    dur_ther = np.array(duration_list)[:, 0]
    dur_ch = np.array(duration_list)[:, 1]

    arr = (np.concatenate([(Mel_scores, Mel_warped_scores, spect_scores,
                            spect_warped_scores, np.array(alpha_formants_list),
                            np.array(response_list),
                           start_t, end_t, dur_ther, start_c,
                           end_c, dur_ch, echo_list)]).T)

    df = pd.DataFrame(arr,
                      columns='Mel_matchin_score Mel_warped_matchin_score spect_score_list spect_score_list_VTLN alpha_formants response_time start_t end_t dur_ther start_c end_c dur_ch echo'.split())

    name = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\mel_and_spect\\DataFrame_all.csv"
    print(f"{name} saved successfully")
    df.to_csv(name)
    np.save("C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\mel_and_spect\\record_arr_all", record_list)
    print("Done :)) ")
