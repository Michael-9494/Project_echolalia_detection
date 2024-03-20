"""
Created on Sun Jul  2 11:49:10 2023

@author: 97254
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
# from numba import jit, guvectorize, int64, float64,cuda
import parselmouth
# from numba import jit, guvectorize, int64, float64,cuda
import scipy
from parselmouth.praat import call
from scipy.interpolate import interp1d
# Setup
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from sklearn.metrics import auc
from tensorflow.keras import backend as K
from sklearn.model_selection import GridSearchCV

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def make_model(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

    input = layers.Input((45, 128, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 4))(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = layers.Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2, 4))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Conv2D(6, (3, 3), padding='same', activation='relu')(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = layers.Conv2D(6, (3, 3), padding='same', activation='relu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    embedding_network = tf.keras.Model(input, x)

    input_1 = layers.Input((45, 128, 1))
    input_2 = layers.Input((45, 128, 1))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = tf.keras.layers.concatenate([tower_1, tower_2])
    # normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    FC1 = layers.Dense(432, activation='relu')(merge_layer)
    Dropout1 = layers.Dropout(0.5)(FC1)
    FC2 = layers.Dense(32, activation='relu')(Dropout1)
    Dropout2 = layers.Dropout(0.5)(FC2)
    # normal_layer2 = tf.keras.layers.BatchNsormalization()(Dropout2)
    output_layer = layers.Dense(1, activation="sigmoid",
                                bias_initializer=output_bias)(Dropout2)

    siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    siamese.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=Adam(learning_rate=1e-4),
                    metrics=METRICS)

    return siamese


# @tf.function
def run_siamese_network(x_train_1, x_train_2, y_train,
                        x_val_1, x_val_2, y_val,
                        x_test_1, x_test_2, y_test,
                        fold_i, save_path, param_grid, VTLN, date):
    neg, pos = np.bincount(y_train)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    initial_bias = np.log([pos/neg])

    input = layers.Input((45, 128, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 4))(x)
    x = layers.Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 4))(x)
    x = layers.Conv2D(6, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(6, (3, 3), padding='same', activation='relu')(x)

    x = layers.Flatten()(x)

    embedding_network = tf.keras.Model(input, x)

    input_1 = layers.Input((45, 128, 1))
    input_2 = layers.Input((45, 128, 1))
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = tf.keras.layers.concatenate([tower_1, tower_2])
    FC1 = layers.Dense(432, activation='relu')(merge_layer)
    Dropout1 = layers.Dropout(0.4)(FC1)
    FC2 = layers.Dense(32, activation='relu')(Dropout1)
    Dropout2 = layers.Dropout(0.4)(FC2)
    output_layer = layers.Dense(1,
                                activation="sigmoid",
                                bias_initializer=tf.keras.initializers.Constant(initial_bias))(Dropout2)

    epochs = param_grid['epochs']
    batch_size = param_grid['batch_size'][0]
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
    optimizer = Adam(learning_rate=1e-3)  # SGD

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    # early_stopping = EarlyStopping(monitor='val_prc', patience=5)

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

    siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    siamese.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=optimizer,
                    metrics=METRICS)

    cls_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    cls_weight_dict = {0: cls_weights[0], 1: cls_weights[1]}

    history = siamese.fit(
        x=[x_train_1, x_train_2],
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[lr_scheduler, early_stopping],
        validation_split=0.0,
        validation_data=([x_val_1, x_val_2], y_val),
        shuffle=True,
        class_weight=cls_weight_dict)

    siamese.save(save_path+"echolalia_detection_" + date+"_" +
                 VTLN+"_" + str(fold_i), overwrite=True, save_format=None)

    results = siamese.evaluate([x_test_1, x_test_2], y_test)

    return results, history, siamese


def plot_ROC(ax, tprs_AUC, aucs_AUC, save_path, save_name, y_lable, x_lable, mean_tpr_1, x, y, title):
    ax.plot(x, y, "k--", label="chance level (AUC = 0.5)")
    mean_fpr = np.linspace(0, 1, 600)
    mean_tpr = np.mean(tprs_AUC, axis=0)
    mean_tpr[-1] = mean_tpr_1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_AUC)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean test ROC (AUC = %0.3f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=1.5,
    )

    std_tpr = np.std(tprs_AUC, axis=0)
    tprs_mel_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_mel_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_mel_lower,
        tprs_mel_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel=x_lable,  # ,"PPV"sensitivity
        ylabel=y_lable,
        title=title,
    )
    ax.axis("square")
    ax.legend(loc='best')
    plt.savefig(save_path+save_name)
    return


def plot_hists(conc_arr, name_1, name_2, hue, echo_logic, no_echo_logic, save_path, Date, title):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    fig.suptitle(title, fontsize=20)

    # Plot the first histogram on the first subplot 'score_echo_sys_2_no_VTLN', multiple="stack"
    sns.histplot(data=conc_arr, x=name_1,
                 bins=50, hue=hue, ax=axs[0], common_norm=False,
                 palette='coolwarm', stat="probability", multiple="stack")
    axs[0].set_xlabel('Matching score', fontsize=12)
    axs[0].set_ylabel('probability', fontsize=12)
    sns.move_legend(axs[0], "center right")
    # Add mean and standard deviation information to the first plot
    mean = conc_arr.loc[echo_logic, name_1].mean()
    std = conc_arr.loc[echo_logic, name_1].std()
    axs[0].axvline(mean,
                   color='r', linestyle='dashed', linewidth=1)
    axs[0].errorbar(mean, 0.05, xerr=std, color='r', linewidth=1, capsize=8)

    axs[0].text(1, 0.95,
                f"Mean (echolalia): {mean:.3f}\nStd (echolalia): {std:.3f}\n\n",
                verticalalignment='top',
                horizontalalignment='right',
                transform=axs[0].transAxes,
                color='r', fontsize=12)

    # sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
    mean = conc_arr.loc[no_echo_logic, name_1].mean()
    std = conc_arr.loc[no_echo_logic, name_1].std()

    axs[0].axvline(mean, color='black', linestyle='dashed', linewidth=1)
    axs[0].errorbar(mean, 0.05, xerr=std, color='black',
                    linewidth=1, capsize=8)

    axs[0].text(1, 0.85, f"Mean (no echolalia): {mean:.3f}\nStd (no echolalia): {std:.3f}\n\n",
                verticalalignment='top',
                horizontalalignment='right',
                transform=axs[0].transAxes,
                color='black', fontsize=12)

    axs[0].set_title("Normal", fontsize=16)

    # Plot the second histogram on the second subplot score_echo_sys_2_VTLN
    sns.histplot(data=conc_arr, x=name_2,
                 bins=50, hue=hue, ax=axs[1], common_norm=False,
                 palette='coolwarm', stat="probability", multiple="stack")
    axs[1].set_xlabel('Matching score VTLN ', fontsize=12)
    axs[1].set_ylabel('probability', fontsize=12)
    # Add mean and standard deviation information to the second plot
    mean = conc_arr.loc[echo_logic, name_2].mean()
    std = conc_arr.loc[echo_logic, name_2].std()

    axs[1].axvline(mean, color='red', linestyle='dashed', linewidth=1)
    axs[1].errorbar(mean, 0.05, xerr=std, color='red', linewidth=1, capsize=8)
    axs[1].text(1, 0.95, f"Mean (echolalia): {mean:.3f}\nStd (echolalia): {std:.3f}\n",
                verticalalignment='top',
                horizontalalignment='right',
                transform=axs[1].transAxes,
                color='red', fontsize=12)
    # sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))
    mean = conc_arr.loc[no_echo_logic, name_2].mean()
    std = conc_arr.loc[no_echo_logic, name_2].std()
    axs[1].axvline(mean, color='black', linestyle='dashed', linewidth=1)
    axs[1].errorbar(mean, 0.05, xerr=std, color='black',
                    linewidth=1, capsize=8)
    axs[1].text(1, 0.85, f"Mean (no echolalia): {mean:.3f}\nStd (no echolalia): {std:.3f}\n",
                verticalalignment='top',
                horizontalalignment='right',
                transform=axs[1].transAxes,
                color='black', fontsize=12)
    sns.move_legend(axs[1], "center right")
    axs[1].set_title("Child VTLN", fontsize=16)
    # plt.savefig(save_path+"histograms_mel_"+Date+".png")
    plt.show()

    return


def make_report(cm_list_UAR, best_UAR_list, best_threshold_list_UAR,
                cm_list_PPV, best_PPV_list, best_PPV_list_from_tresh,
                best_threshold_list_PPV, save_path, save_name):
    data = {
        'TP_UAR': [item[1, 1] for item in cm_list_UAR],
        'FP_UAR': [item[0, 1] for item in cm_list_UAR],
        'FN_UAR': [item[1, 0] for item in cm_list_UAR],
        'TN_UAR': [item[0, 0] for item in cm_list_UAR],
        'best_UAR_list': best_UAR_list,
        'best_threshold_list_UAR': best_threshold_list_UAR,
        'TP_PPV': [item[1, 1] for item in cm_list_PPV],
        'FP_PPV': [item[0, 1] for item in cm_list_PPV],
        'FN_PPV': [item[1, 0] for item in cm_list_PPV],
        'TN_PPV': [item[0, 0] for item in cm_list_PPV],
        'best_PPV_list': best_PPV_list,
        'best_PPV_list_from_tresh': best_PPV_list_from_tresh,
        'best_threshold_list_PPV': best_threshold_list_PPV
    }

    df = pd.DataFrame(data)

    # Write DataFrame to Excel
    df.to_excel(save_path+save_name, index=False)

    print(f"Saved: {save_name} ")

    return


def showResults(test, pred):
    target_names = ['No Echolalia', 'Echolalia']
    print(classification_report(test, pred, target_names=target_names))
    accuracy = accuracy_score(test, pred)
    precision = precision_score(test, pred, average='weighted')
    f1Score = f1_score(test, pred, average='weighted')
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    cm = confusion_matrix(test, pred)
    print(cm)
    return (accuracy, precision, f1Score, cm)


def find_stuff(predictions, y_test):
    # # np.linspace(1,3,4)
    ThRange = np.linspace(1, 0, 600)

    # initialize the False Positive (Pf) and True Positive (Pd) lists
    one_minus_specificity = np.zeros(len(ThRange))
    sensitivity = np.zeros(len(ThRange))
    PPV = np.zeros(len(ThRange))
    UAR = np.zeros(len(ThRange))
    f1 = np.zeros(len(ThRange))

    # calculate Pf and Pd for each threshold value.loc[echo_logic, "Mel_matchin_score"]
    for i, Th in enumerate(ThRange):
        # Ntp = np.sum() #we said echolalia and it is true
        y_pred_classes_test = (predictions > Th).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes_test).ravel()
        sensitivity[i] = (tp/(tp+fn))
        one_minus_specificity[i] = (fp/(fp+tn))
        if tp + fp != 0:
            PPV[i] = (tp / (tp + fp))
        else:
            PPV[i] = 0.0
        UAR[i] = 0.5*((tp/(tp+fn))+(tn/(tn+fp)))
        # print("Weighted Average Recall:", weighted_recall)

        # F1 Score
        f1[i] = f1_score(y_test, y_pred_classes_test)

    auc_mel = np.trapz(sensitivity, one_minus_specificity)
    auc_ppv_test = np.trapz(PPV, sensitivity)

    best_threshold_UAR = ThRange[np.argmax(UAR)]
    best_threshold_PPV = ThRange[np.argmax(PPV)]

    Predictions = (predictions > 0.5).astype(int)

    predictions_UAR = (predictions > best_threshold_UAR).astype(int)
    predictions_PPV = (predictions > best_threshold_PPV).astype(int)

    return (sensitivity,
            one_minus_specificity,
            PPV, UAR, ThRange, auc_mel,
            auc_ppv_test,
            predictions_UAR, predictions_PPV, Predictions,
            best_threshold_UAR, best_threshold_PPV)


def find_stuff_sys_1(echo, no_echo, y, y_test):

    # # combine the two lists and get the minimum and maximum threshold values
    Uni_mel = np.concatenate([echo, no_echo.to_numpy()])
    MinTh = min(Uni_mel)
    MaxTh = max(Uni_mel)

    # # np.linspace(1,3,4)
    ThRange = np.linspace(MinTh, MaxTh+1, 600)

    # initialize the False Positive (Pf) and True Positive (Pd) lists
    one_minus_specificity = np.zeros(len(ThRange))
    sensitivity = np.zeros(len(ThRange))
    PPV = np.zeros(len(ThRange))
    UAR = np.zeros(len(ThRange))
    f1 = np.zeros(len(ThRange))

    # calculate Pf and Pd for each threshold value.loc[echo_logic, "Mel_matchin_score"]
    for i, Th in enumerate(ThRange):
        # Ntp = np.sum() #we said echolalia and it is true
        y_pred_classes_test = (y < Th).astype(int)

        TN, FP, FN, TP = confusion_matrix(y_test, y_pred_classes_test).ravel()
        sensitivity[i] = (TP/(TP+FN))  # recall TPR
        one_minus_specificity[i] = (FP/(FP+TN))  # False positive Rate

        if TP + FP != 0:
            PPV[i] = (TP / (TP + FP))
        else:
            PPV[i] = 0.0
        UAR[i] = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
        # print("Weighted Average Recall:", weighted_recall)

        # F1 Score
        f1[i] = f1_score(y_test, y_pred_classes_test)

    auc = np.trapz(sensitivity, one_minus_specificity)
    auc_ppv_test = np.trapz(PPV, sensitivity)

    best_threshold_UAR = ThRange[np.argmax(UAR)]
    best_threshold_PPV = ThRange[np.argmax(PPV)]

    predictions_UAR = (y < best_threshold_UAR).astype(int)
    predictions_PPV = (y < best_threshold_PPV).astype(int)

    # classifier_output = np.concatenate([background,signal])
    # true_value = np.concatenate([np.zeros_like(background, dtype=int), np.ones_like(signal, dtype=int)])

    return (sensitivity,
            one_minus_specificity,
            PPV, UAR, ThRange, auc,
            auc_ppv_test,
            predictions_UAR, predictions_PPV,
            best_threshold_UAR, best_threshold_PPV)


def plt_metric(history, metric, title, save_path, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    fig9, ax9 = plt.subplots(figsize=(6, 6))
    ax9.plot(history[metric])
    if has_valid:
        ax9.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.savefig(save_path+metric+'_'+str(title)+'.png')
    # plt.show()


def scheduler(epoch, lr):
    if epoch % 3 == 0 and epoch != 0:
        lr = lr * 0.1
    return lr


def segment_analysis_ver_2(Sound, fbank, window_length=0.04, time_step=0.02, nfilt=69, mean_seg_num=200, flag=0):

    spectrogram_for_mel = Sound.to_spectrogram(window_length=window_length,
                                               time_step=time_step,
                                               maximum_frequency=8000)

    spect_plot = 10 * np.log10(spectrogram_for_mel.values)
    mel = (fbank @ spectrogram_for_mel.values)
    # # Numerical Stability spect_T
    mel = np.where((mel == 0), np.finfo(float).eps, mel)
    mel = 10 * np.log10(mel)  # convert to dB

    X = resample(mel, int(mean_seg_num), axis=1)
    spect = resample(spect_plot, int(mean_seg_num), axis=1)
    # print:
    if flag == 1:
        # plt.figure(figsize=(14, 12))
        # plt.imshow(spect,origin='lower')
        # plt.show()
        plt.figure(figsize=(14, 12))
        plt.imshow(X, origin='lower')
        plt.show()
    return X, spect


def compute_filterbank(low_freq_mel, NFFT, nfilt, Fs):
    high_freq_mel = (2595 * np.log10(1 + (Fs / 2) / 700))  # Convert Hz to Mel
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    # We don't have the frequency resolution required to put filters at the exact points calculated above,
    # so we need to round those frequencies to the nearest FFT bin.
    # This process does not affect the accuracy of the features.
    # To convert the frequncies to fft bin numbers we need to know the FFT size and the sample rate,
    binn = np.floor((NFFT + 1) * hz_points / Fs)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(binn[m - 1])   # left
        f_m = int(binn[m])             # center
        f_m_plus = int(binn[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - binn[m - 1]) / (binn[m] - binn[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (binn[m + 1] - k) / (binn[m + 1] - binn[m])
    filter_banks = fbank
    return filter_banks


def segment_analysis(Sound, fbank, f0min=75, f0max=1600, window_length=0.04, time_step=0.02, nfilt=69, flag=0):

    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pitch = call(Sound, "To Pitch (cc)", 0,  # time_step(s)0
                 f0min,  # f0min pitch_floor(Hz)
                 15,
                 'no',
                 0.02,  # silence_threshold
                 0.45,  # Voicing_th
                 0.01,  # Octave_cost
                 0.35,  # octave-Jump cost
                 0.25,  # Voiced/Un-Voiced 0.14
                 f0max)  # f0max
    # pitch = snd_part.to_pitch()
    # pitch_T[pitch_values==0] = np.nan
    # Ifdesired,pre-emphasizethesounADOSragmentbeforecalculatingthespectrogram    spectrogram=pre_emphasized_snd.to_spectrogram(window_length=0.03,maximum_ ˓→frequency=8000)
    pre_emphasized_snd = Sound.copy()
    pre_emphasized_snd.pre_emphasize()
    spectt = pre_emphasized_snd.to_spectrogram(window_length=window_length,
                                               time_step=time_step,
                                               maximum_frequency=8000)
    spectrogram_for_mel = pre_emphasized_snd.to_spectrogram(window_length=window_length,
                                                            time_step=time_step,
                                                            maximum_frequency=8000)

    t, f = spectrogram_for_mel.x_grid(), spectrogram_for_mel.y_grid()
    f = f[1:]
    ff_min = 100
    ff_max = 6000
    m_high = (2595 * np.log10(1 + (ff_max) / 700))
    m_min = (2595 * np.log10(1 + (ff_min) / 700))
    high_freq_mel = (2595 * np.log10(1 + (16000 / 2) / 700)
                     )  # Convert Hz to Mel
    # Equally spaced in Mel scale
    mel_points = np.linspace(0, high_freq_mel, nfilt + 2)[1:-1]
    logic_mel = np.array(((mel_points >= m_min) & (mel_points <= m_high)).T)

    logic_spect = np.array((f >= ff_min) & (f <= ff_max))

    pointProcess = call(Sound, "To PointProcess (periodic, cc)",
                        f0min,
                        f0max)
    numPoints = call(pointProcess, "Get number of points")

    formants = call(Sound, "To Formant (burg)",
                    time_step,    # time step(s),
                    5,
                    # formant ceiling(Hz),
                    6000,
                    # window length(s),
                    window_length/2,
                    50)

    f1_listT = []
    f2_listT = []
    f3_listT = []
    time_fT = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        time_fT.append(t)
        f1_listT.append(f1)
        f2_listT.append(f2)
        f3_listT.append(f3)

    x_F = np.array([f1_listT, f2_listT, f3_listT])
    mu = np.nanmean(x_F, axis=1)

    intensityT = Sound.to_intensity()  # intensityT.values
    X_P = np.atleast_2d(pitch.selected_array['frequency'])  # Sound.values
    # X_P = Preprocess.normalize_feature_sequence( X_P,norm='2')
    # Y_P1 = (pitchC.selected_array['frequency'])

    X_P = np.where((X_P == 0), np.nan, X_P)  # Numerical Stability
    mu_p = np.nanmean(X_P)

    # X_P[X_P==0] =20
    # X_P[X_P!=20] =5
    duration = Sound.xs()[-1]

    spect_for_mel = spectrogram_for_mel.values
    spect = (spectt.values)
    spect = 10 * np.log10(np.where((spect == 0), np.finfo(float).eps, spect))
   # X_S_T =  (Preprocess.ltw(X_S_T.T, mean_seg_num)).T

    mel = (fbank @ spect_for_mel)
    # # Numerical Stability spect_T
    mel = np.where((mel == 0), np.finfo(float).eps, mel)

    mel = 10 * np.log10(mel)  # convert to dB
    # create DCT matrix
    (m, k) = np.mgrid[0:nfilt, 0:nfilt]
    m = m+1  # % m [1...M=nfilt]
    lamba_m = (2*m-1)/(2*nfilt)
    DCT_mat = np.sqrt(2 / nfilt) * np.cos(np.pi * lamba_m * k)
    DCT_mat[0, :] = DCT_mat[0, :] / np.sqrt(2)
    # A = np.round(DCT_mat@DCT_mat.T)
    # mfcc =DCT_mat@ mel

    X = mel[logic_mel, :]
    spect_ret = spect[logic_spect, :]

    return X.astype("float32"), spect_ret.astype("float32"), duration, mu_p, mu


def segment_analysis_VTLN(Sound, fbank, mu_T, f0min=75, f0max=1600, window_length=0.04, time_step=0.02, nfilt=69, alpha_3=1, flag=0):

    #     # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pitch = call(Sound, "To Pitch (cc)", 0,  # time_step(s)0
                 f0min,  # f0min pitch_floor(Hz)
                 15,
                 'no',
                 0.02,  # silence_threshold
                 0.45,  # Voicing_th
                 0.01,  # Octave_cost
                 0.35,  # octave-Jump cost
                 0.25,  # Voiced/Un-Voiced 0.14
                 f0max)  # f0max
    # pitch = snd_part.to_pitch()
    # pitch_T[pitch_values==0] = np.nan
    # Ifdesired,pre-emphasizethesounADOSragmentbeforecalculatingthespectrogram    spectrogram=pre_emphasized_snd.to_spectrogram(window_length=0.03,maximum_ ˓→frequency=8000)
    pre_emphasized_snd = Sound.copy()
    pre_emphasized_snd.pre_emphasize()
    spectt = pre_emphasized_snd.to_spectrogram(window_length=window_length,
                                               time_step=time_step,
                                               maximum_frequency=8000)
    spectrogram_for_mel = pre_emphasized_snd.to_spectrogram(window_length=window_length,
                                                            time_step=time_step,
                                                            maximum_frequency=8000)

    t, f = spectrogram_for_mel.x_grid(), spectrogram_for_mel.y_grid()
    f = f[1:]
    ff_min = 100
    ff_max = 6000
    m_high = (2595 * np.log10(1 + (ff_max) / 700))
    m_min = (2595 * np.log10(1 + (ff_min) / 700))
    high_freq_mel = (2595 * np.log10(1 + (16000 / 2) / 700)
                     )  # Convert Hz to Mel
    # Equally spaced in Mel scale
    mel_points = np.linspace(0, high_freq_mel, nfilt + 2)[1:-1]
    logic_mel = np.array(((mel_points >= m_min) & (mel_points <= m_high)).T)

    logic_spect = np.array((f >= ff_min) & (f <= ff_max))

    pointProcess = call(Sound, "To PointProcess (periodic, cc)",
                        f0min,
                        f0max)
    numPoints = call(pointProcess, "Get number of points")

    formants = call(Sound, "To Formant (burg)",
                    time_step,                                  # time step(s),
                    5,
                    # formant ceiling(Hz),
                    6000,
                    # window length(s),
                    window_length/2,
                    50)                                         # Pre-emphasis from(Hz)

    f1_list = []
    f2_list = []
    f3_list = []
    time_f = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        time_f.append(t)
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    x_F = np.array([f1_list, f2_list, f3_list])
    mu_C = np.nanmean(x_F, axis=1)

    intensityT = Sound.to_intensity()  # intensityT.values
    X_P = np.atleast_2d(pitch.selected_array['frequency'])  # Sound.values
    # X_P = Preprocess.normalize_feature_sequence( X_P,norm='2')
    # Y_P1 = (pitchC.selected_array['frequency'])

    X_P = np.where((X_P == 0), np.nan, X_P)  # Numerical Stability
    mu_p = np.nanmean(X_P)

    duration = Sound.xs()[-1]

    spect_for_mel = spectrogram_for_mel.values
    spect = (spectt.values)
    spect = 10 * np.log10(np.where((spect == 0), np.finfo(float).eps, spect))
    spect_ret = spect[logic_spect, :]

    mel = (fbank @ spect_for_mel)
    # # Numerical Stability spect_T
    mel = np.where((mel == 0), np.finfo(float).eps, mel)

    mel = 10 * np.log10(mel)  # convert to dB

    (m, k) = np.mgrid[0:nfilt, 0:nfilt]
    m = m+1  # % m [1...M=nfilt]
    lamba_m = (2*m-1)/(2*nfilt)
    DCT_mat = np.sqrt(2 / nfilt) * np.cos(np.pi * lamba_m * k)
    DCT_mat[0, :] = DCT_mat[0, :] / np.sqrt(2)
    A = np.round(DCT_mat@DCT_mat.T)
    # mfcc =DCT_mat@ mel

    X = mel[logic_mel, :]

    num_bins, num_frames = spect_for_mel.shape
    freq_bins = np.arange(num_bins)

    alpha = mu_T/mu_C
    alpha_3 = alpha[2]  # 1.1

    new_freq_bins = freq_bins*alpha_3
    f = interp1d(freq_bins, spect_for_mel, kind='nearest', axis=0,
                 bounds_error=False, fill_value='extrapolate')
    spectrogram_warped_for_mel = f(new_freq_bins)

    # spectrogram_warped
    num_bins, num_frames = spect.shape
    freq_bins = np.arange(num_bins)

    new_freq_bins = freq_bins*alpha_3
    f = interp1d(freq_bins, spect, kind='nearest', axis=0,
                 bounds_error=False, fill_value='extrapolate')
    spectrogram_warped = (f(new_freq_bins))
    spectrogram_warped_ret = spectrogram_warped[logic_spect, :]

    filter_banks_C_warped = (fbank @ spectrogram_warped_for_mel)
    # # Numerical Stability
    # Fb_C = np.where( ( filter_banks_C <= (filter_banks_C.max()- 20)), np.finfo(float).eps, filter_banks_C)

    mel_spectrogram_wa = 10 * np.log10(filter_banks_C_warped)
    # mfcc_wa =DCT_mat@ mel_spectrogram_wa
    X_warped = mel_spectrogram_wa[logic_mel, :]

    return X.astype("float32"), X_warped.astype("float32"), spect_ret.astype("float32"), spectrogram_warped_ret.astype("float32"), duration, mu_p, mu_C, alpha_3


# @jit(nopython=True,target_backend='cuda')
def compute_accumulated_cost_matrix(C):
    """Compute the accumulated cost matrix given the cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D


# @jit(nopython=True,target_backend='cuda')
def compute_optimal_warping_path(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix

    Returns:
        P (np.ndarray): Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])

            if val == D[n-1, m-1]:  # Match
                cell = (n-1, m-1)

            elif val == D[n-1, m]:  # Insertion
                cell = (n-1, m)

            else:
                cell = (n, m-1)

        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)


# @jit( forceobj=True,target_backend='cuda')
def dynamic_time_warping(x, y, metric="cosine"):

    # Cost matrix
    Cost_mat = scipy.spatial.distance.cdist(x.T, y.T, metric)
    N, M = Cost_mat.shape

    # accumulated Cost matrix
    acc_cost_mat = compute_accumulated_cost_matrix(Cost_mat)

    # find optimal path
    path = compute_optimal_warping_path(acc_cost_mat)

    return(path, acc_cost_mat, Cost_mat, (acc_cost_mat[-1, -1])/(M+N))
