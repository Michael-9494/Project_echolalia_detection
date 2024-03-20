# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:28:36 2023

@author: polonik
"""


# Setup

import Functions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)


# tf.config.set_visible_devices([], 'GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
print(tf.test.is_built_with_cuda())
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    VTLNs = ["no_VTLN", "VTLN"]   # Identifier for VTLN process #
    date = "27_Jul_2023"  # Date for the current run

    # save_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\Results\\"
    # read_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\Results\\"

    target_names = ['No Echolalia', 'Echolalia']  # Names of the target classes

    mean_fpr = np.linspace(0, 1, 600)  # Mean False Positive Rate

    # Paths for saving and reading data
    # Path to child label data in Excel format
    # "D:\\Autism\\Echolalia_proj_Michael\\report\\All_Data_26_04_2023.xlsx"
    all_child_label = "D:\\Recs_echolalia_26_04_2023\\All_Data_26_04_2023.xlsx"
    save_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\Results\\"+date+"\\"
    read_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\mel_and_spect\\"

    # Load information about the database and child labels
    # "D:\\Autism\\Echolalia_proj_Michael\\DATA"
    Path = "D:\\Recs_echolalia_26_04_2023\\all"  # Path to the database
    Database = os.listdir(Path)  # List of files in the database directory

    # Read data from Excel file
    all_data = pd.read_excel(all_child_label, sheet_name='Sheet1', index_col=(
        None), dtype={'Child_Key': int})

    Echo_score = (all_data['IM_Echo'])  # Echo scores

    date_record_1 = all_data['Record_date']  # Record dates
    # Convert the Timestamp objects to the desired format
    converted_dates = date_record_1.dt.strftime("%d%m%y")
    # Create Child_Key by concatenating Child_Key_1 with date
    Child_Key_1 = (all_data['Child_Key'])
    Child_Key = [str(Child_Key_1) + '_' + str(date)
                 for Child_Key_1, date in zip(Child_Key_1, converted_dates)]
    Child_Key = pd.DataFrame(Child_Key)

    y_train_0 = np.load(read_path+"echo_list_all.npy")
    y_train_0_pd = pd.DataFrame(y_train_0)
    record_arr_all = np.load(read_path+"record_arr_all.npy")
    record_arr_all_p = pd.DataFrame(
        record_arr_all, columns='record'.split())

    for VTLN in VTLNs:
        print(f'{VTLN}')

        # Load and reshape training data
        Mels_train = np.load(read_path+"mel_"+VTLN + ".npy")
        Mels_train = np.reshape(Mels_train, (Mels_train.shape[0],
                                             Mels_train.shape[2],
                                             Mels_train.shape[3],
                                             Mels_train.shape[1]))

        # Initialize lists to store various metrics and predictions for different folds
        (tprs_AUC_train, aucs_AUC_train, tprs_PPV_train, aucs_PPV_train,
         best_UAR_list_train, best_PPV_list_train, best_PPV_list_from_tresh_train,
         best_threshold_list_UAR_train, best_threshold_list_PPV_train,
         score_echo_train, predictions_list_train, predictions_list_pred_UAR_train,
         predictions_list_pred_PPV_train,
         predictions_list_rec_train, cm_list_UAR_train, cm_list_PPV_train,
         True_echo_list_train
         ) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        (tprs_AUC_val, aucs_AUC_val, tprs_PPV_val, aucs_PPV_val, best_UAR_list_val,
         best_PPV_list_val, best_PPV_list_from_tresh_val,
         best_threshold_list_UAR_val, best_threshold_list_PPV_val, score_echo_val,
         predictions_list_val, predictions_list_pred_UAR_val,
         predictions_list_pred_PPV_val, predictions_list_rec_val, cm_list_UAR_val,
         cm_list_PPV_val, True_echo_list_val
         ) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        (tprs_AUC, aucs_AUC, tprs_PPV, aucs_PPV, best_prediction_list, report_list,
         best_threshold_list_UAR, best_threshold_list_PPV, best_UAR_list,
         best_PPV_list_from_tresh, best_PPV_list, cm_list_UAR, cm_list_PPV,
         predictions_list_test, predictions_list_pred_UAR,
         predictions_list_pred_PPV, predictions_list_rec, score_echo,
         True_echo_list_test
         ) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        (fold_list_test, fold_list_val, fold_list_train) = [], [], []
        # Initialize StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        skf1 = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)

        # Create subplots for visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        fig_train, ax_train = plt.subplots(figsize=(6, 6))
        fig2_train, ax2_train = plt.subplots(figsize=(6, 6))
        fig_val, ax_val = plt.subplots(figsize=(6, 6))
        fig2_val, ax2_val = plt.subplots(figsize=(6, 6))
        # Iterate over folds for cross-validation
        for fold_i, (train_val_index, test_index) in enumerate(skf.split(Child_Key, Echo_score)):
            # Split data into training/validation and test sets for this fold
            train_val_data, test_data = Child_Key.iloc[train_val_index], Child_Key.iloc[test_index]
            # Further split training/validation data into training and validation sets
            train_idx, val_idx = next(skf1.split(X=train_val_data.reset_index(drop=True),
                                                 y=Echo_score.iloc[train_val_index].reset_index(drop=True)))
            train_data, val_data = (train_val_data.reset_index(drop=True).loc[train_idx],
                                    train_val_data.reset_index(drop=True).loc[val_idx])

            print(f"Fold {fold_i}:")
            logic_record_all_train = record_arr_all_p['record'].isin(
                train_data[0])

            x_train_1 = Mels_train[logic_record_all_train, :, :, 0]
            x_train_2 = Mels_train[logic_record_all_train, :, :, 1]
            y_train = y_train_0[logic_record_all_train]
            y_train_record_name = record_arr_all_p['record'][logic_record_all_train]

            num_echo_train = sum(y_train)
            num_no_echo_train = len(x_train_1)-num_echo_train
            # print(f"Train: {train_data}\n")
            print('num_echo_train  '+f"{num_echo_train}")
            print('num_no_echo_train  '+f"{num_no_echo_train}")

            logic_record_all_val = record_arr_all_p['record'].isin(val_data[0])
            x_val_1 = Mels_train[logic_record_all_val, :, :, 0]
            x_val_2 = Mels_train[logic_record_all_val, :, :, 1]
            y_val = y_train_0[logic_record_all_val]
            y_val_record_name = record_arr_all_p['record'][logic_record_all_val]

            num_echo_val = sum(y_val)
            num_no_echo_val = len(x_val_1)-num_echo_val
            # print(f"Val: {val_data}\n")
            print('num_echo_val  '+f"{num_echo_val}")
            print('num_no_echo_val  '+f"{num_no_echo_val}")

            logic_record_all_test = record_arr_all_p['record'].isin(
                test_data[0])
            x_test_1 = Mels_train[logic_record_all_test, :, :, 0]
            x_test_2 = Mels_train[logic_record_all_test, :, :, 1]
            y_test = y_train_0[logic_record_all_test]
            y_test_record_name = record_arr_all_p['record'][logic_record_all_test]

            num_echo_test = sum(y_test)
            num_no_echo_test = len(x_test_1)-num_echo_test
            # print(f"Test: {test_data}\n")
            print('num_echo_test  '+f"{num_echo_test}")
            print('num_no_echo_test  '+f"{num_no_echo_test}")

            batch_size = [1024, 128, 512, 256]
            epochs = 100

            param_grid = dict(batch_size=batch_size, epochs=epochs)
            # y_train = np.array(y_train)
            # y_val = np.array(y_val)
            # y_test = np.array(y_test)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_prc',
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True)

            siamese = Functions.make_model()
            results = siamese.evaluate(
                [x_test_1[:10], x_test_2[:10]], y_test[:10],
                batch_size=param_grid['batch_size'][0], verbose=0)

            print("Loss: {:0.4f}".format(results[0]))

            neg, pos = np.bincount(y_train)
            total = neg + pos
            print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
                total, pos, 100 * pos / total))
            initial_bias = np.log([pos/neg])
# initial_bias

            predictions = siamese.predict([x_test_1, x_test_2])
            score_echo.append(predictions)
            True_echo_list_test.append(y_test)
            True_echo_list_val.append(y_val)
            True_echo_list_train.append(y_train)

            Functions.plt_metric(history.history, "auc",
                                 "Model_AUC_"+"fold_"+str(fold_i)+"_" +
                                 str(VTLN),  save_path)
            Functions.plt_metric(history.history, "accuracy",
                                 "Model_accuracy_"+"fold_"+str(fold_i)+"_" +
                                 str(VTLN), save_path)
            Functions.plt_metric(history.history, "loss",
                                 "Binary_cross-entropy_loss_"+"fold_" +
                                 str(fold_i)+"_"+str(VTLN), save_path)

            (sensitivity, one_minus_specificity,
             PPV, UAR, ThRange, AUC,
             auc_ppv_test, predictions_UAR,
             predictions_PPV, Predictions,
             best_threshold_UAR,
             best_threshold_PPV) = Functions.find_stuff(predictions, y_test)

            print("from 0.5 \n")
            Functions.showResults(y_test, Predictions)

            print("from PPV \n")
            Functions.showResults(y_test, predictions_PPV)

            interp_tpr_AUC = np.interp(
                mean_fpr, one_minus_specificity, sensitivity)
            interp_tpr_AUC[0] = 0.0
            tprs_AUC.append(interp_tpr_AUC)
            aucs_AUC.append(AUC)
            ax.plot(one_minus_specificity, sensitivity, lw=1.2, alpha=0.4,
                    label=f" ROC fold {fold_i} (AUC={AUC:.3f}) ")

            interp_tpr_PPV = np.interp(mean_fpr, sensitivity, PPV)
            interp_tpr_PPV[0] = 0.0
            tprs_PPV.append(interp_tpr_PPV)
            aucs_PPV.append(auc_ppv_test)
            ax2.plot(sensitivity, PPV, lw=1.2, alpha=0.4,
                     label=f" ROC fold {fold_i} (AUC={auc_ppv_test:.3f}) ")

            # plt.plot(Pd,Pf)
            # find the optimal threshold

            best_UAR_list.append(UAR[np.argmax(UAR)])
            best_PPV_list.append(PPV[np.argmax(PPV)])
            print("from UAR \n")
            (accuracy, precision, f1Score, cm) = Functions.showResults(
                y_test, predictions_UAR)

            predictions_list_test.append(y_train_0_pd[logic_record_all_test])
            predictions_list_pred_UAR.append(predictions_UAR)
            predictions_list_pred_PPV.append(predictions_PPV)
            predictions_list_rec.append(y_test_record_name)
            fold_vec = [fold_i] * len(y_test_record_name)

            fold_list_test.append(fold_vec)
            report_list.append((accuracy, precision, f1Score, cm))

            cm_list_UAR.append(confusion_matrix(y_test, predictions_UAR))
            cm_list_PPV.append(confusion_matrix(y_test, predictions_PPV))

            best_PPV_list_from_tresh.append(PPV[np.argmax(UAR)])

            best_threshold_list_UAR.append(best_threshold_UAR)
            best_threshold_list_PPV.append(best_threshold_PPV)

            '''
            train
            '''
            predictions_train = siamese.predict([x_train_1, x_train_2])
            score_echo_train.append(predictions_train)

            (sensitivity_train, one_minus_specificity_train,
             PPV_train, UAR_train, ThRange_train, AUC_train,
             auc_ppv_train, predictions_UAR_train,
             predictions_PPV_train, Predictions_train,
             best_threshold_UAR_train,
             best_threshold_PPV_train) = Functions.find_stuff(predictions_train, y_train)

            interp_tpr_AUC_train = np.interp(
                mean_fpr, one_minus_specificity_train, sensitivity_train)
            interp_tpr_AUC_train[0] = 0.0
            tprs_AUC_train.append(interp_tpr_AUC_train)
            aucs_AUC_train.append(AUC_train)
            ax_train.plot(one_minus_specificity_train, sensitivity_train, lw=1.2,
                          alpha=0.4, label=f" ROC fold {fold_i} (AUC={AUC_train:.3f}) ")

            interp_tpr_PPV_train = np.interp(
                mean_fpr, sensitivity_train, PPV_train)
            interp_tpr_PPV_train[0] = 0.0
            tprs_PPV_train.append(interp_tpr_PPV_train)
            aucs_PPV_train.append(auc_ppv_train)
            ax2_train.plot(sensitivity_train, PPV_train, lw=1.2, alpha=0.4,
                           label=f" ROC fold {fold_i} (AUC={auc_ppv_train:.3f}) ")

            # plt.plot(Pd,Pf)
            # find the optimal threshold

            best_UAR_list_train.append(UAR_train[np.argmax(UAR_train)])
            best_PPV_list_train.append(PPV_train[np.argmax(PPV_train)])

            predictions_list_train.append(y_train_0_pd[logic_record_all_train])
            predictions_list_pred_UAR_train.append(predictions_UAR_train)
            predictions_list_pred_PPV_train.append(predictions_PPV_train)
            predictions_list_rec_train.append(y_train_record_name)
            fold_vec = [fold_i] * len(y_train_record_name)

            fold_list_train.append(fold_vec)

            cm_list_UAR_train.append(confusion_matrix(
                y_train, predictions_UAR_train))
            cm_list_PPV_train.append(confusion_matrix(
                y_train, predictions_PPV_train))

            best_PPV_list_from_tresh_train.append(
                PPV_train[np.argmax(UAR_train)])

            best_threshold_list_UAR_train.append(best_threshold_UAR_train)
            best_threshold_list_PPV_train.append(best_threshold_PPV_train)

            '''
            validation
            '''
            predictions_val = siamese.predict([x_val_1, x_val_2])
            score_echo_val.append(predictions_val)
            (sensitivity_val, one_minus_specificity_val,
             PPV_val, UAR_val, ThRange_val, AUC_val,
             auc_ppv_val, predictions_UAR_val,
             predictions_PPV_val, Predictions_val,
             best_threshold_UAR_val,
             best_threshold_PPV_val) = Functions.find_stuff(predictions_val, y_val)

            interp_tpr_AUC_val = np.interp(
                mean_fpr, one_minus_specificity_val, sensitivity_val)
            interp_tpr_AUC_val[0] = 0.0
            tprs_AUC_val.append(interp_tpr_AUC_val)
            aucs_AUC_val.append(AUC_val)
            ax_val.plot(one_minus_specificity_val, sensitivity_val, lw=1.2,
                        alpha=0.4, label=f" ROC fold {fold_i} (AUC={AUC_val:.3f})")

            interp_tpr_PPV_val = np.interp(
                mean_fpr, sensitivity_val, PPV_val)
            interp_tpr_PPV_val[0] = 0.0
            tprs_PPV_val.append(interp_tpr_PPV_val)
            aucs_PPV_val.append(auc_ppv_val)
            ax2_val.plot(sensitivity_val, PPV_val, lw=1.2, alpha=0.4,
                         label=f" ROC fold {fold_i} (AUC={auc_ppv_val:.3f}) ")

            # plt.plot(Pd,Pf)
            # find the optimal threshold

            best_UAR_list_val.append(UAR_val[np.argmax(UAR_val)])
            best_PPV_list_val.append(PPV_val[np.argmax(PPV_val)])

            predictions_list_val.append(y_train_0_pd[logic_record_all_train])
            predictions_list_pred_UAR_val.append(predictions_UAR_val)
            predictions_list_pred_PPV_val.append(predictions_PPV_val)
            predictions_list_rec_val.append(y_val_record_name)
            fold_vec = [fold_i] * len(y_val_record_name)
            fold_list_val.append(fold_vec)

            cm_list_UAR_val.append(confusion_matrix(
                y_val, predictions_UAR_val))
            cm_list_PPV_val.append(confusion_matrix(
                y_val, predictions_PPV_val))

            best_PPV_list_from_tresh_val.append(PPV_val[np.argmax(UAR_val)])

            best_threshold_list_UAR_val.append(best_threshold_UAR_val)
            best_threshold_list_PPV_val.append(best_threshold_PPV_val)

            K.clear_session()

        '''
        save test results
        '''
        concatenated_pred_UAR = pd.concat([
            pd.Series(predictions_list_pred_UAR[0].flatten()),
            pd.Series(predictions_list_pred_UAR[1].flatten()),
            pd.Series(predictions_list_pred_UAR[2].flatten()),
            pd.Series(predictions_list_pred_UAR[3].flatten()),
            pd.Series(predictions_list_pred_UAR[4].flatten())
        ], axis=0)
        concatenated_pred_PPV = pd.concat([
            pd.Series(predictions_list_pred_PPV[0].flatten()),
            pd.Series(predictions_list_pred_PPV[1].flatten()),
            pd.Series(predictions_list_pred_PPV[2].flatten()),
            pd.Series(predictions_list_pred_PPV[3].flatten()),
            pd.Series(predictions_list_pred_PPV[4].flatten())
        ], axis=0)
        concatenated_rec = pd.concat([predictions_list_rec[0],
                                     predictions_list_rec[1],
                                     predictions_list_rec[2],
                                     predictions_list_rec[3],
                                     predictions_list_rec[4]], axis=0)

        concatenated_score_echo = np.concatenate([(score_echo[0]),
                                                  (score_echo[1]),
                                                  (score_echo[2]),
                                                  (score_echo[3]),
                                                  (score_echo[4])], axis=0)

        concatenated_True_echo_test = np.concatenate([(True_echo_list_test[0]),
                                                      (True_echo_list_test[1]),
                                                      (True_echo_list_test[2]),
                                                      (True_echo_list_test[3]),
                                                      (True_echo_list_test[4])],
                                                     axis=0)
        concatenated_score_fold = np.concatenate([(fold_list_test[0]),
                                                  (fold_list_test[1]),
                                                  (fold_list_test[2]),
                                                  (fold_list_test[3]),
                                                  (fold_list_test[4])], axis=0)
        conc_arr = pd.DataFrame()
        conc_arr['record'] = concatenated_rec.values
        conc_arr['True_echo'] = concatenated_True_echo_test
        conc_arr['prediction_sys_2_'+VTLN +
                 '_UAR'] = concatenated_pred_UAR.values
        conc_arr['prediction_sys_2_'+VTLN +
                 '_PPV'] = concatenated_pred_PPV.values

        conc_arr['score_echo_sys_2_'+VTLN] = concatenated_score_echo
        conc_arr['fold_test_sys_2_'+VTLN] = concatenated_score_fold
        conc_arr.index = concatenated_rec.index
        conc_arr = conc_arr.sort_index()
        save_name = "concatenated_sys_2_test_" + VTLN+".csv"
        conc_arr.to_csv(save_path+save_name)

        save_name = "ROC_curve_sys_2_test_sensitivity_" + VTLN
        Functions.plot_ROC(ax, tprs_AUC, aucs_AUC, save_path, save_name,
                           y_lable="sensitivity", x_lable="1-Specificity",
                           mean_tpr_1=1.0, x=[1, 0], y=[1, 0], title=" test ROC " + VTLN)

        save_name = "ROC_curve_sys_2_test_PPV_" + VTLN
        Functions.plot_ROC(ax2, tprs_PPV, aucs_PPV, save_path, save_name,
                           y_lable="PPV", x_lable="sensitivity",
                           mean_tpr_1=0.0, x=[1, 0], y=[1, 0], title=" test ROC " + VTLN)
        # plt.show()

        save_name = "output_sys_2_"+VTLN+".xlsx"
        Functions.make_report(cm_list_UAR, best_UAR_list, best_threshold_list_UAR,
                              cm_list_PPV, best_PPV_list, best_PPV_list_from_tresh,
                              best_threshold_list_PPV, save_path, save_name)
        # siamese.summary()
        '''
        save train results
        '''
        concatenated_pred_UAR_train = pd.concat([
            pd.Series(predictions_list_pred_UAR_train[0].flatten()),
            pd.Series(predictions_list_pred_UAR_train[1].flatten()),
            pd.Series(predictions_list_pred_UAR_train[2].flatten()),
            pd.Series(predictions_list_pred_UAR_train[3].flatten()),
            pd.Series(predictions_list_pred_UAR_train[4].flatten())
        ], axis=0)
        concatenated_pred_PPV_train = pd.concat([
            pd.Series(predictions_list_pred_PPV_train[0].flatten()),
            pd.Series(predictions_list_pred_PPV_train[1].flatten()),
            pd.Series(predictions_list_pred_PPV_train[2].flatten()),
            pd.Series(predictions_list_pred_PPV_train[3].flatten()),
            pd.Series(predictions_list_pred_PPV_train[4].flatten())
        ], axis=0)
        concatenated_rec_train = pd.concat([predictions_list_rec_train[0],
                                            predictions_list_rec_train[1],
                                            predictions_list_rec_train[2],
                                            predictions_list_rec_train[3],
                                            predictions_list_rec_train[4]],
                                           axis=0)

        concatenated_score_echo_train = np.concatenate([(score_echo_train[0]),
                                                        (score_echo_train[1]),
                                                        (score_echo_train[2]),
                                                        (score_echo_train[3]),
                                                        (score_echo_train[4])],
                                                       axis=0)
        concatenated_score_fold_train = np.concatenate([(fold_list_train[0]),
                                                        (fold_list_train[1]),
                                                        (fold_list_train[2]),
                                                        (fold_list_train[3]),
                                                        (fold_list_train[4])],
                                                       axis=0)
        concatenated_True_echo_train = np.concatenate([(True_echo_list_train[0]),
                                                      (True_echo_list_train[1]),
                                                      (True_echo_list_train[2]),
                                                      (True_echo_list_train[3]),
                                                      (True_echo_list_train[4])],
                                                      axis=0)

        conc_arr_train = pd.DataFrame()

        conc_arr_train['record_train'] = concatenated_rec_train.values
        conc_arr_train['True_echo_train'] = concatenated_True_echo_train
        conc_arr_train['prediction_train_sys_2_'+VTLN +
                       '_UAR'] = concatenated_pred_UAR_train.values

        conc_arr_train['prediction_train_sys_2_'+VTLN +
                       '_PPV'] = concatenated_pred_PPV_train.values

        conc_arr_train['fold_train_sys_2_' +
                       VTLN] = concatenated_score_fold_train

        conc_arr_train['score_train_echo_sys_2_' +
                       VTLN] = concatenated_score_echo_train

        conc_arr_train.index = concatenated_rec_train.index
        conc_arr_train = conc_arr_train.sort_index()

        save_name = "concatenated_train_sys_2_" + VTLN+".csv"
        conc_arr_train.to_csv(save_path+save_name)

        save_name = "ROC_curve_train_sys_2_sensitivity_" + VTLN
        Functions.plot_ROC(ax_train, tprs_AUC_train, aucs_AUC_train, save_path,
                           save_name,
                           y_lable="sensitivity", x_lable="1-Specificity",
                           mean_tpr_1=1.0, x=[1, 0], y=[1, 0], title=" train ROC "
                           + VTLN)

        save_name = "ROC_curve_train_sys_2_PPV_" + VTLN
        Functions.plot_ROC(ax2_train, tprs_PPV_train, aucs_PPV_train, save_path,
                           save_name,
                           y_lable="PPV", x_lable="sensitivity",
                           mean_tpr_1=0.0, x=[1, 0], y=[1, 0], title=" train ROC "
                           + VTLN)
        # plt.show()

        save_name = "output_train_sys_2_"+VTLN+".xlsx"
        Functions.make_report(cm_list_UAR_train, best_UAR_list_train,
                              best_threshold_list_UAR_train,
                              cm_list_PPV_train, best_PPV_list_train,
                              best_PPV_list_from_tresh_train,
                              best_threshold_list_PPV_train, save_path, save_name)
        '''
        save validation results
        '''
        concatenated_pred_UAR_val = pd.concat([
            pd.Series(predictions_list_pred_UAR_val[0].flatten()),
            pd.Series(predictions_list_pred_UAR_val[1].flatten()),
            pd.Series(predictions_list_pred_UAR_val[2].flatten()),
            pd.Series(predictions_list_pred_UAR_val[3].flatten()),
            pd.Series(predictions_list_pred_UAR_val[4].flatten())
        ], axis=0)
        concatenated_pred_PPV_val = pd.concat([
            pd.Series(predictions_list_pred_PPV_val[0].flatten()),
            pd.Series(predictions_list_pred_PPV_val[1].flatten()),
            pd.Series(predictions_list_pred_PPV_val[2].flatten()),
            pd.Series(predictions_list_pred_PPV_val[3].flatten()),
            pd.Series(predictions_list_pred_PPV_val[4].flatten())
        ], axis=0)
        concatenated_rec_val = pd.concat([predictions_list_rec_val[0],
                                         predictions_list_rec_val[1],
                                         predictions_list_rec_val[2],
                                         predictions_list_rec_val[3],
                                         predictions_list_rec_val[4]],
                                         axis=0)

        concatenated_score_echo_val = np.concatenate([(score_echo_val[0]),
                                                      (score_echo_val[1]),
                                                      (score_echo_val[2]),
                                                      (score_echo_val[3]),
                                                      (score_echo_val[4])],
                                                     axis=0)
        concatenated_score_fold_val = np.concatenate([(fold_list_val[0]),
                                                      (fold_list_val[1]),
                                                      (fold_list_val[2]),
                                                      (fold_list_val[3]),
                                                      (fold_list_val[4])],
                                                     axis=0)
        concatenated_True_echo_val = np.concatenate([(True_echo_list_val[0]),
                                                     (True_echo_list_val[1]),
                                                     (True_echo_list_val[2]),
                                                     (True_echo_list_val[3]),
                                                     (True_echo_list_val[4])],
                                                    axis=0)
        conc_arr_val = pd.DataFrame()

        conc_arr_val['record_val'] = concatenated_rec_val.values
        conc_arr_val['True_echo_val'] = concatenated_True_echo_val
        conc_arr_val['prediction_val_sys_2_'+VTLN +
                     '_UAR'] = concatenated_pred_UAR_val.values
        conc_arr_val['prediction_val_sys_2_'+VTLN +
                     '_PPV'] = concatenated_pred_PPV_val.values
        conc_arr_val['fold_val_sys_2_'+VTLN] = concatenated_score_fold_val
        conc_arr_val['score_val_echo_sys_2_' +
                     VTLN] = concatenated_score_echo_val
        conc_arr_val.index = concatenated_rec_val.index
        conc_arr_val = conc_arr_val.sort_index()
        save_name = "concatenated_val_sys_2_" + VTLN+".csv"
        conc_arr_val.to_csv(save_path+save_name)

        save_name = "ROC_curve_val_sys_2_sensitivity_" + VTLN+"_"+date
        Functions.plot_ROC(ax_val, tprs_AUC_val, aucs_AUC_val, save_path,
                           save_name,
                           y_lable="sensitivity", x_lable="1-Specificity",
                           mean_tpr_1=1.0, x=[1, 0], y=[1, 0], title=" val ROC " + VTLN)

        save_name = "ROC_curve_val_sys_2_PPV_" + VTLN+"_"+date
        Functions.plot_ROC(ax2_val, tprs_PPV_val, aucs_PPV_val, save_path,
                           save_name,
                           y_lable="PPV", x_lable="sensitivity",
                           mean_tpr_1=0.0, x=[1, 0], y=[1, 0], title=" val ROC " + VTLN)

        save_name = "output_val_sys_2_"+VTLN+"_"+date+".xlsx"
        Functions.make_report(cm_list_UAR_val, best_UAR_list_val,
                              best_threshold_list_UAR_val,
                              cm_list_PPV_val, best_PPV_list_val,
                              best_PPV_list_from_tresh_val,
                              best_threshold_list_PPV_val, save_path, save_name)
plt.show()
