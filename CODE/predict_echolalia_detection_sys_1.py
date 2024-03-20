
import Functions
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\97254\anaconda3\envs\Project_speech_MP')
sys.path.append(
    r'C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\py\py-part')


# @jit( forceobj=True)
if __name__ == '__main__':
    target_names = ['No Echolalia', 'Echolalia']
    date = "02_Aug_2023"  # Date for the current run

    read_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\mel_and_spect\\"
    save_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\Results\\"+date+"\\"

    VTLNs = ["no_VTLN", "VTLN"]
    df_train = pd.read_csv(read_path+"DataFrame_all.csv")
    # Select columns to scale

    # Update the 'echo' column values
    df_train['echo'] = df_train['echo']
    record_arr_all = np.load(read_path+"record_arr_all.npy")
    df_train['record'] = record_arr_all
    df_train_all = df_train.copy()
    columns = ' response_time start_t end_t dur_ther start_c end_c dur_ch record'.split()
    df_train_all.drop(columns, axis=1, inplace=True)
    # Update the 'echo' column values

    Path = r'D:\Recs_echolalia_26_04_2023\all'
    Database = os.listdir(Path)
    all_child_label = "D:\Recs_echolalia_26_04_2023\All_Data_26_04_2023.xlsx"

    all_data = pd.read_excel(all_child_label, sheet_name='Sheet1', index_col=(
        None), dtype={'Child_Key': int})
    # print(all_data.info())

    # log =((all_data["Child_Key"]) == ( int(Database[3].split("_")[0]) ))
    Echo_score = (all_data['IM_Echo'])

    date_record_1 = all_data['Record_date']
    # Convert the Timestamp objects to the desired format
    converted_dates = date_record_1.dt.strftime("%d%m%y")

    Child_Key_1 = (all_data['Child_Key'])
    Child_Key = [str(Child_Key_1) + '_' + str(date)
                 for Child_Key_1, date in zip(Child_Key_1, converted_dates)]
    Child_Key = pd.DataFrame(Child_Key)

    conc_arr = pd.DataFrame()
    for VTLN in VTLNs:
        print(f'{VTLN}')

        (best_threshold_list_UAR, best_threshold_list_PPV, best_UAR_list,
         best_PPV_list_from_tresh, best_PPV_list, tprs_AUC, aucs_AUC,
         tprs_PPV, aucs_PPV, cm_list_UAR, cm_list_PPV, predictions_list_test,
         predictions_list_pred, predictions_list_pred_PPV, predictions_list_rec
         ) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        skf1 = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        for fold_i, (train_val_index, test_index) in enumerate(skf.split(Child_Key, Echo_score)):
            train_val_data, test_data = Child_Key.iloc[train_val_index], Child_Key.iloc[test_index]

            train_idx, val_idx = next(skf1.split(X=train_val_data.reset_index(drop=True),
                                                 y=Echo_score.iloc[train_val_index].reset_index(drop=True)))
            train_data, val_data = (train_val_data.reset_index(drop=True).loc[train_idx],
                                    train_val_data.reset_index(drop=True).loc[val_idx])

            print(f"Fold {fold_i}:")

            # Create a logical vector based on the recrord_name column in df_aug
            # logic_vector_aug = df_aug['record_name'].isin(train_data[0])
            logic_vector_train = df_train['record'].isin(train_data[0])
            logic_vector_test = df_train['record'].isin(test_data[0])

            # Extract the rows from df_aug based on the logical vector
            # train_data_df_aug = df_aug_all[logic_vector_aug]

            # Extract the rows from df_aug based on the logical vector
            train_data_df = df_train_all[logic_vector_train]
            # pd.concat([, train_data_df_aug])
            df_train_with_aug = train_data_df

            y_train = df_train_with_aug['echo']

            echo_logic_train = (df_train_with_aug['echo'] == 1)
            no_echo_logic_train = (df_train_with_aug['echo'] == 0)

            df_test = df_train_all[logic_vector_test]
            y_test = df_test['echo']
            y_test_log = y_test.map
            echo_logic_test = (df_test['echo'] == 1)

            no_echo_logic_test = (df_test['echo'] == 0)
            y_test_record_name = df_train['record'][logic_vector_test]

            if VTLN == "no_VTLN":

                X_train_no_echo = df_train_with_aug['Mel_matchin_score'][no_echo_logic_train]
                X_train_echo = df_train_with_aug['Mel_matchin_score'][echo_logic_train]
                w = "Mel_matchin_score"
            else:
                X_train_no_echo = df_train_with_aug['Mel_warped_matchin_score'][no_echo_logic_train]
                X_train_echo = df_train_with_aug['Mel_warped_matchin_score'][echo_logic_train]
                w = "Mel_warped_matchin_score"

            (sensitivity, one_minus_specificity, PPV, UAR, ThRange, AUC,
             auc_ppv_test, predictions_UAR,
             predictions_PPV, best_threshold_UAR,
             best_threshold_PPV) = Functions.find_stuff_sys_1(X_train_echo,
                                                              X_train_no_echo,
                                                              df_test[w],
                                                              y_test)

            print("from UAR \n")
            Functions.showResults(y_test, predictions_UAR)

            print("from PPV \n")
            Functions.showResults(y_test, predictions_PPV)

            interp_tpr_AUC = np.interp(
                np.linspace(0, 1, 600), one_minus_specificity, sensitivity)
            interp_tpr_AUC[0] = 0.0
            # print(f"interp_tpr {interp_tpr}\n")

            tprs_AUC.append(interp_tpr_AUC)

            aucs_AUC.append(AUC)
            ax.plot(one_minus_specificity, sensitivity, lw=1.2, alpha=0.4,
                    label=f" ROC fold {fold_i} (AUC = {AUC:.3f}) ")

            interp_tpr_PPV = np.interp(
                np.linspace(0, 1, 600), sensitivity, PPV)
            interp_tpr_PPV[0] = 0.0
            tprs_PPV.append(interp_tpr_PPV)
            aucs_PPV.append(auc_ppv_test)
            ax2.plot(sensitivity, PPV, lw=1.2, alpha=0.4,
                     label=f" PRC fold {fold_i} (AUPRC = {auc_ppv_test:.3f}) ")

            # plt.plot(Pd,Pf)
            # find the optimal threshold

            best_UAR_list.append(UAR[np.argmax(UAR)])
            best_PPV_list.append(PPV[np.argmax(PPV)])
            print("from UAR \n")
            (accuracy, precision, f1Score, cm) = Functions.showResults(
                y_test, predictions_UAR)

            predictions_list_test.append(y_test)
            predictions_list_pred.append(predictions_UAR)
            predictions_list_pred_PPV.append(predictions_PPV)
            predictions_list_rec.append(y_test_record_name)
            # report_list.append((accuracy, precision, f1Score,cm))

            cm_list_UAR.append(confusion_matrix(y_test, predictions_UAR))
            cm_list_PPV.append(confusion_matrix(y_test, predictions_PPV))

            best_PPV_list_from_tresh.append(PPV[np.argmax(UAR)])

            best_threshold_list_UAR.append(best_threshold_UAR)
            best_threshold_list_PPV.append(best_threshold_PPV)

    #######################################################################################
        ''' 
        save part
        '''
    #######################################################################################

        concatenated_test = pd.concat([predictions_list_test[0],
                                       predictions_list_test[1],
                                       predictions_list_test[2],
                                       predictions_list_test[3],
                                       predictions_list_test[4]],
                                      axis=0).sort_index()
        concatenated_pred_UAR = pd.concat([predictions_list_pred[0],
                                           predictions_list_pred[1],
                                           predictions_list_pred[2],
                                           predictions_list_pred[3],
                                           predictions_list_pred[4]],
                                          axis=0).sort_index()
        concatenated_pred_PPV = pd.concat([predictions_list_pred_PPV[0],
                                           predictions_list_pred_PPV[1],
                                           predictions_list_pred_PPV[2],
                                           predictions_list_pred_PPV[3],
                                           predictions_list_pred_PPV[4]],
                                          axis=0).sort_index()
        concatenated_rec = pd.concat([predictions_list_rec[0],
                                      predictions_list_rec[1],
                                      predictions_list_rec[2],
                                      predictions_list_rec[3],
                                      predictions_list_rec[4]],
                                     axis=0).sort_index()
        # conc_arr = pd.concat(
        #     [concatenated_test, concatenated_pred_UAR, concatenated_pred_PPV,
        #      concatenated_rec], axis=1)

        conc_arr['record'] = concatenated_rec.values
        conc_arr['True_echo'] = concatenated_test.values
        conc_arr['prediction_sys_1_'+VTLN +
                 '_UAR'] = concatenated_pred_UAR.values
        conc_arr['prediction_sys_1_'+VTLN +
                 '_PPV'] = concatenated_pred_PPV.values

        save_name = "ROC_curve_sys_1_test_" + VTLN
        Functions.plot_ROC(ax, tprs_AUC, aucs_AUC, save_path, save_name,
                           y_lable="sensitivity", x_lable="1-Specificity",
                           mean_tpr_1=1.0, x=[1, 0], y=[1, 0],
                           title=" test ROC " + VTLN)
        save_name = "PRC_sys_1_VTLN_test_" + VTLN
        Functions.plot_ROC(ax2, tprs_PPV, aucs_PPV, save_path, save_name,
                           y_lable="PPV", x_lable="sensitivity",
                           mean_tpr_1=0.0, x=[1, 0], y=[1, 0],
                           title=" test PRC " + VTLN)

        save_name = "output_sys_1_"+VTLN+".xlsx"
        Functions.make_report(cm_list_UAR, best_UAR_list, best_threshold_list_UAR,
                              cm_list_PPV, best_PPV_list, best_PPV_list_from_tresh,
                              best_threshold_list_PPV, save_path, save_name)

conc_arr.index = concatenated_rec.index
conc_arr = conc_arr.sort_index()
save_name = "concatenated_sys_1_test.csv"
conc_arr.to_csv(save_path+save_name)
plt.show()
