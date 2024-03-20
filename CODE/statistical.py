# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:16:08 2023

@author: 97254
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import Functions
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import scipy.stats as stats
import pingouin as pg


def wilcoxon_test(a, b):
    print(f"\n\n{a.name} VS {b.name}")
  # Assume 'true_values' and 'predicted_values' are arrays of the true and predicted values for a specific variable
    statistic, p_value = wilcoxon(a, b)
    print("wilcoxon")
    # Interpret the results
    if p_value < 0.05:
        print(
            f"statistic:{statistic}\np_value:{p_value}\nThe median difference between true and predicted values is statistically significant.")
    else:
        print(
            f"statistic:{statistic}\np_value:{p_value}\nThere is no significant difference between true and predicted values.")
    statistic, p_value = mannwhitneyu(a, b)
    # Interpret the results
    print("\nmannwhitneyu")
    if p_value < 0.05:
        print(
            f"statistic:{statistic}\np_value:{p_value}\nThe distributions of true and predicted values are significantly different.")
    else:
        print(f"statistic:{statistic}\np_value:{p_value}\nThere is no significant difference between the distributions of true and predicted values.")


def polt_reg(df1, df2, df):
    print("\n\n"+df1)
    print(df2)
    df = df.copy()
    df.columns = df.columns.astype(str)  # Convert column names to strings
    # Calculate regression metrics
    y_true = df[df1]
    y_pred = df[df2]
    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)
    corr = y_true.corr(y_pred)
    correa_pearsonr, p_pearsonr = pearsonr(y_true, y_pred)

    print('pearsonr correlation coefficient: %.3f' % correa_pearsonr)
    # interpret the significance
    alpha = 0.05
    if p_pearsonr > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' %
              p_pearsonr)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p_pearsonr)

    # Scatter plot with regression metrics
    ax = sns.regplot(x=df1, y=df2, data=df)
    plt.setp(ax.collections[1], alpha=0)
    plt.xlabel("Actual Echolalia rate")
    plt.ylabel("Estimated Echolalia rate")

    plt.title(f"{df1} VS {df2}")
    plt.text(0.5, 0.95,
             f"pearson: {correa_pearsonr:.3f}, p={p_pearsonr:.3f}",
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.show()


if __name__ == '__main__':

    Date = "02_Aug_2023"  # "29_Jul_2023"
    read_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\mel_and_spect\\"
    save_path = "C:\\Users\\97254\\Documents\\GitHub\\Project-speech\\Project_echolalia_detection\\py\\py-part\\CODE\\Results\\"+Date+"\\"

    all_child_label = "All_Data_26_04_2023.xlsx"

    all_data = pd.read_excel(all_child_label, sheet_name='Sheet1', index_col=(
        None), dtype={'Child_Key': int})
    # all_data['Birth date']
    # all_data['Age_(months)'] = (current_date - all_data['Birth date']) // datetime.timedelta(days=30)

    date_record_1 = all_data['Record_date']
    # Convert the Timestamp objects to the desired format
    converted_dates = date_record_1.dt.strftime("%d%m%y")
    Child_Key_1 = all_data['Child_Key']

    Child_Key = [str(Child_Key_1) + '_' + str(date)
                 for Child_Key_1, date in zip(Child_Key_1, converted_dates)]
    Child_Key = pd.DataFrame(Child_Key)
    all_data.index = Child_Key.values
    all_data = all_data.sort_index()

    data_analize = all_data[['IM_Echo', 'ADOS', 'RRB', 'Age_m']]
    sns.barplot(data=all_data, x='ADOS', y='RRB', hue='IM_Echo')
    plt.show()
    sns.pairplot(data=all_data,
                 hue='IM_Echo',
                 vars=' Age_m ADOS RRB '.split(),
                 palette='coolwarm',
                 markers=["o", "s", "D"])
    plt.show()
    sns.countplot(x='Gender', data=all_data, palette='coolwarm')
    plt.show()

    all_data['Age_m'].mean()
    all_data['Age_m'].std()
    all_data['ADOS'].mean()
    all_data['ADOS'].std()
    desc_all_data = all_data.describe()

    df_train = pd.read_csv(read_path+"DataFrame_all.csv")
    df_train = df_train.drop('Unnamed: 0', axis=1)
    palette = sns.color_palette('coolwarm')

    # Update the 'echo' column values
    df_train['echo'] = df_train['echo']
    record_arr_all = np.load(read_path+"record_arr_all.npy")
    df_train['record'] = record_arr_all
    echo_train = (df_train['echo'] == 1)
    no_echo_train = (df_train['echo'] == 0)
    desc_df_train = df_train.describe()
    df_train.columns

    sns.histplot(data=df_train, x="alpha_formants",
                 bins=50, hue="echo", common_norm=False,
                 palette='coolwarm', stat="probability", multiple="stack")

    Functions.plot_hists(df_train, 'Mel_matchin_score',
                         'Mel_warped_matchin_score',
                         "echo", echo_train,
                         no_echo_train,
                         save_path, Date, "sys_1")
    # Functions.plot_hists(df_train, 'Mel_matchin_score',
    #                      'Mel_warped_matchin_score',
    #                      "echo", echo_train,
    #                      no_echo_train,
    #                      save_path, Date, "sys_1")

    sns.countplot(x=df_train['echo'], palette='coolwarm')
    plt.xlabel('Echolalia')
    plt.ylabel('Count')

    # Add text annotations to the bars
    total_count = df_train['echo'].value_counts()
    for i, count in enumerate(total_count):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # Create a custom legend with consistent colors
    legend_labels = ['No Echolalia', 'Echolalia']
    legend_handles = [plt.bar(0, 0, color=palette[0]),
                      plt.bar(0, 0, color=palette[4])]
    plt.legend(legend_handles, legend_labels)

    plt.show()

    # conc_arr.info()
    conc_arr = pd.read_csv(save_path+"concatenated_sys_2_test.csv")
    conc_arr = conc_arr.drop('Unnamed: 0', axis=1)
    True_echo = conc_arr['True_echo']

    # Calculate means
    echo_logic = (conc_arr['True_echo'] == 1)
    no_echo_logic = (conc_arr['True_echo'] == 0)

    title = "sys_2_Test"
    Functions.plot_hists(conc_arr, 'score_echo_sys_2_no_VTLN',
                         'score_echo_sys_2_VTLN',
                         "True_echo", echo_logic, no_echo_logic,
                         save_path, Date, title)
    conc_arr.columns
    Functions.plot_hists(conc_arr, 'score_echo_sys_2_no_VTLN',
                         'score_echo_sys_2_VTLN',
                         "fold_test_sys_2_no_VTLN", echo_logic,
                         no_echo_logic,
                         save_path, Date, title)
    a = conc_arr["True_echo"]
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_test_sys_2_no_VTLN"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_2_no_VTLN_UAR"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_2_no_VTLN_PPV"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_test_sys_2_VTLN"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_2_VTLN_UAR"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_2_VTLN_PPV"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_1_no_VTLN_UAR"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_1_no_VTLN_PPV"])
    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_1_VTLN_UAR"])

    wilcoxon_test(conc_arr["True_echo"],
                  conc_arr["prediction_sys_1_VTLN_PPV"])

    stats.ttest_ind(
        a=conc_arr["True_echo"], b=conc_arr["prediction_test_sys_2_no_VTLN"],
        equal_var=True)

    # Conducting two-sample ttest
    result = pg.mwu(
        conc_arr["True_echo"], conc_arr["prediction_test_sys_2_no_VTLN"], alternative='two-sided')
    # Print the result
    print(result)
    result = pg.ttest(conc_arr["True_echo"],
                      conc_arr["prediction_sys_2_no_VTLN_UAR"],
                      correction=True)
    # Print the result
    print(result)
    result = pg.ttest(conc_arr["True_echo"],
                      conc_arr["prediction_sys_2_no_VTLN_PPV"],
                      correction=True)
    # Print the result
    print(result)
    conc_arr_train = pd.read_csv(save_path+"concatenated_sys_2_train.csv")
    conc_arr_train = conc_arr_train.drop('Unnamed: 0', axis=1)
    # Calculate means
    echo_logic_train = (conc_arr_train['True_echo_train'] == 1)
    no_echo_logic_train = (conc_arr_train['True_echo_train'] == 0)
    title = "sys_2_Train"
    Functions.plot_hists(conc_arr_train, 'score_train_echo_sys_2_no_VTLN',
                         'score_train_echo_sys_2_VTLN',
                         "True_echo_train", echo_logic_train,
                         no_echo_logic_train,
                         save_path, Date, title)

    # Functions.plot_hists(conc_arr_train, 'score_train_echo_sys_2_no_VTLN',
    #                      'score_train_echo_sys_2_VTLN',
    #                      "aug_list_train_VTLN", echo_logic_train,
    #                      no_echo_logic_train,
    #                      save_path, Date, title)
    conc_arr_train.columns
    Functions.plot_hists(conc_arr_train, 'score_train_echo_sys_2_no_VTLN',
                         'score_train_echo_sys_2_VTLN',
                         "fold_train_sys_2_no_VTLN", echo_logic_train,
                         no_echo_logic_train,
                         save_path, Date, title)
    title = "sys_2_Validation"
    conc_arr_val = pd.read_csv(save_path+"concatenated_sys_2_val.csv")
    conc_arr_val = conc_arr_val.drop('Unnamed: 0', axis=1)
    # Calculate means
    echo_logic_val = (conc_arr_val['True_echo_val'] == 1)
    no_echo_logic_val = (conc_arr_val['True_echo_val'] == 0)
    Functions.plot_hists(conc_arr_val, 'score_val_echo_sys_2_no_VTLN',
                         'score_val_echo_sys_2_VTLN',
                         "True_echo_val", echo_logic_val,
                         no_echo_logic_val,
                         save_path, Date, title)

    conc_arr.columns
    Functions.plot_hists(conc_arr_val, 'score_val_echo_sys_2_no_VTLN',
                         'score_val_echo_sys_2_VTLN',
                         "fold_val_sys_2_no_VTLN", echo_logic_val,
                         no_echo_logic_val,
                         save_path, Date, title)

    grouped = conc_arr.groupby('record')
    echolalia_sum_true = grouped['True_echo'].sum()
    non_echolalia_sum_true = grouped.size() - grouped['True_echo'].sum()

    echolalia_sum_pred_sys_1 = grouped['prediction_sys_1_no_VTLN_UAR'].sum()
    echolalia_sum_pred_sys_1_VTLN = grouped['prediction_sys_1_VTLN_UAR'].sum()
    echolalia_sum_pred_sys_2 = grouped['prediction_sys_2_no_VTLN_UAR'].sum()
    echolalia_sum_pred_sys_2_VTLN = grouped['prediction_sys_2_VTLN_UAR'].sum()

    non_echolalia_sum_pred_sys_1 = grouped.size(
    ) - grouped['prediction_sys_1_no_VTLN_UAR'].sum()
    non_echolalia_sum_pred_sys_1_VTLN = grouped.size(
    ) - grouped['prediction_sys_1_VTLN_UAR'].sum()
    non_echolalia_sum_pred_sys_2 = grouped.size(
    ) - grouped['prediction_sys_2_no_VTLN_UAR'].sum()
    non_echolalia_sum_pred_sys_2_VTLN = grouped.size(
    ) - grouped['prediction_sys_2_VTLN_UAR'].sum()

    grouped_df = pd.concat([(echolalia_sum_true/grouped.size()),
                            # 1021315504_090820
                            (echolalia_sum_pred_sys_1/grouped.size()),
                            (echolalia_sum_pred_sys_1_VTLN/grouped.size()),
                            (echolalia_sum_pred_sys_2/grouped.size()),
                            (echolalia_sum_pred_sys_2_VTLN/grouped.size())],
                           axis=1)
    grouped_df.columns = ['Echolalia_Rate_True',
                          'Echolalia_Rate_sys_1_UAR',
                          'Echolalia_Rate_sys_1_VTLN_UAR',
                          'Echolalia_Rate_sys_2_UAR',
                          'Echolalia_Rate_sys_2_VTLN_UAR']

    # Generate a mask for the upper triangle

    data_analize.index = grouped_df.index
    # print(grouped_df)

    all_dataframe = pd.concat([data_analize, grouped_df], axis=1)
    # all_dataframe.info()

    grouped_ppv = conc_arr.groupby('record')
    echolalia_sum_true_ppv = grouped_ppv['True_echo'].sum()
    non_echolalia_sum_true_ppv = grouped_ppv.size() - \
        grouped_ppv['True_echo'].sum()

    echolalia_sum_pred_sys_1_ppv = grouped_ppv['prediction_sys_1_no_VTLN_PPV'].sum(
    )
    echolalia_sum_pred_sys_1_VTLN_ppv = grouped_ppv['prediction_sys_1_VTLN_PPV'].sum(
    )
    echolalia_sum_pred_sys_2_ppv = grouped_ppv['prediction_sys_2_no_VTLN_PPV'].sum(
    )
    echolalia_sum_pred_sys_2_VTLN_ppv = grouped_ppv['prediction_sys_2_VTLN_PPV'].sum(
    )

    non_echolalia_sum_pred_sys_1_ppv = grouped_ppv.size(
    ) - grouped_ppv['prediction_sys_1_no_VTLN_PPV'].sum()
    non_echolalia_sum_pred_sys_1_VTLN_ppv = grouped_ppv.size(
    ) - grouped_ppv['prediction_sys_1_VTLN_PPV'].sum()
    non_echolalia_sum_pred_sys_2_ppv = grouped_ppv.size(
    ) - grouped_ppv['prediction_sys_2_no_VTLN_PPV'].sum()
    non_echolalia_sum_pred_sys_2_VTLN_ppv = grouped_ppv.size(
    ) - grouped_ppv['prediction_sys_2_VTLN_PPV'].sum()

    grouped_df_ppv = pd.concat([(echolalia_sum_true_ppv/grouped_ppv.size()),
                                # 1021315504_090820
                                (echolalia_sum_pred_sys_1_ppv/grouped_ppv.size()),
                                (echolalia_sum_pred_sys_1_VTLN_ppv/grouped_ppv.size()),
                                (echolalia_sum_pred_sys_2_ppv/grouped_ppv.size()),
                                (echolalia_sum_pred_sys_2_VTLN_ppv/grouped_ppv.size())],
                               axis=1)
    grouped_df_ppv.columns = ['Echolalia_Rate_True',
                              'Echolalia_Rate_sys_1_PPV',
                              'Echolalia_Rate_sys_1_VTLN_PPV',
                              'Echolalia_Rate_sys_2_PPV',
                              'Echolalia_Rate_sys_2_VTLN_PPV']

    # Generate a mask for the upper triangle

    data_analize.index = grouped_df_ppv.index
    # print(grouped_df_ppv)

    all_dataframe_ppv = pd.concat([data_analize, grouped_df_ppv], axis=1)
    # all_dataframe_ppv.info()

    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_1_UAR', all_dataframe)
    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_1_VTLN_UAR', all_dataframe)
    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_2_UAR', all_dataframe)
    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_2_VTLN_UAR', all_dataframe)

    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_1_PPV', all_dataframe_ppv)
    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_1_VTLN_PPV', all_dataframe_ppv)
    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_2_PPV', all_dataframe_ppv)
    polt_reg('Echolalia_Rate_True',
             'Echolalia_Rate_sys_2_VTLN_PPV', all_dataframe_ppv)

    kappa_1_no_vtln_to_vtln_uar = cohen_kappa_score(
        conc_arr['prediction_sys_1_VTLN_UAR'], conc_arr['prediction_sys_1_no_VTLN_UAR'])
    # Form confusion matrix for McNemar's test
    table1_no_vtln_to_vtln = confusion_matrix(
        conc_arr['prediction_sys_1_VTLN_UAR'], conc_arr['prediction_sys_1_no_VTLN_UAR'])
    # Perform McNemar's test
    result_1_no_vtln_to_vtln_uar = mcnemar(table1_no_vtln_to_vtln, exact=True)
    print(
        f'1_no_vtln_to_vtln_uar McNemar\'s test p-value : {result_1_no_vtln_to_vtln_uar.pvalue}')
    print("Cohen's Kappa between kappa_1_no_vtln_to_vtln_uar :",
          kappa_1_no_vtln_to_vtln_uar)

    kappa_1_no_vtln_to_vtln_ppv = cohen_kappa_score(
        conc_arr['prediction_sys_1_VTLN_PPV'], conc_arr['prediction_sys_1_no_VTLN_PPV'])
    # Form confusion matrix for McNemar's test
    table1_no_vtln_to_vtln_ppv = confusion_matrix(
        conc_arr['prediction_sys_1_VTLN_PPV'], conc_arr['prediction_sys_1_no_VTLN_PPV'])
    # Perform McNemar's test
    result_1_no_vtln_to_vtln_ppv = mcnemar(
        table1_no_vtln_to_vtln_ppv, exact=True)
    print(
        f'1_no_vtln_to_vtln_ppv McNemar\'s test p-value: {result_1_no_vtln_to_vtln_ppv.pvalue}')
    print("Cohen's Kappa between kappa_1_no_vtln_to_vtln_ppv :",
          kappa_1_no_vtln_to_vtln_uar)

    kappa_2_no_vtln_to_2_vtln_uar = cohen_kappa_score(
        conc_arr['prediction_sys_2_VTLN_UAR'], conc_arr['prediction_sys_2_no_VTLN_UAR'])
    # Form confusion matrix for McNemar's test
    table_2_no_vtln_to_2_vtln_uar = confusion_matrix(
        conc_arr['prediction_sys_2_VTLN_UAR'], conc_arr['prediction_sys_2_no_VTLN_UAR'])
    # Perform McNemar's test
    result_2_no_vtln_to_2_vtln_uar = mcnemar(
        table_2_no_vtln_to_2_vtln_uar, exact=True)
    print(
        f'2_no_vtln_to_2_vtln_uar McNemar\'s test p-value: {result_2_no_vtln_to_2_vtln_uar.pvalue}')
    print("Cohen's Kappa between  kappa_2_no_vtln_to_2_vtln_uar:",
          kappa_2_no_vtln_to_2_vtln_uar)

    kappa_2_no_vtln_to_2_vtln_ppv = cohen_kappa_score(
        conc_arr['prediction_sys_2_VTLN_PPV'], conc_arr['prediction_sys_2_no_VTLN_PPV'])
    # Form confusion matrix for McNemar's test
    table_2_no_vtln_to_2_vtln_ppv = confusion_matrix(
        conc_arr['prediction_sys_2_VTLN_PPV'], conc_arr['prediction_sys_2_no_VTLN_PPV'])
    # Perform McNemar's test
    result_2_no_vtln_to_2_vtln_ppv = mcnemar(
        table_2_no_vtln_to_2_vtln_ppv, exact=True)
    print(
        f'2_no_vtln_to_2_vtln_ppv McNemar\'s test p-value: {result_2_no_vtln_to_2_vtln_ppv.pvalue}')
    print("Cohen's Kappa between  kappa_2_no_vtln_to_2_vtln_ppv:",
          kappa_2_no_vtln_to_2_vtln_uar)

    kappa_1_vtln_to_2_vtln_uar = cohen_kappa_score(
        conc_arr['prediction_sys_1_VTLN_UAR'], conc_arr['prediction_sys_2_VTLN_UAR'])
    table_1_vtln_to_2_vtln_uar = confusion_matrix(
        conc_arr['prediction_sys_1_VTLN_UAR'], conc_arr['prediction_sys_2_VTLN_UAR'])
    result_1_vtln_to_2_vtln_uar = mcnemar(
        table_1_vtln_to_2_vtln_uar, exact=True)
    print(
        f'1_vtln_to_2_vtln_uar McNemar\'s test p-value: {result_1_vtln_to_2_vtln_uar.pvalue}')
    print("Cohen's Kappa kappa_1_vtln_to_2_vtln_uar:", kappa_1_vtln_to_2_vtln_uar)

    kappa_1_vtln_to_2_vtln_ppv = cohen_kappa_score(
        conc_arr['prediction_sys_1_VTLN_PPV'], conc_arr['prediction_sys_2_VTLN_PPV'])
    table_1_vtln_to_2_vtln_ppv = confusion_matrix(
        conc_arr['prediction_sys_1_VTLN_PPV'], conc_arr['prediction_sys_2_VTLN_PPV'])
    result_1_vtln_to_2_vtln_ppv = mcnemar(
        table_1_vtln_to_2_vtln_ppv, exact=True)
    print(
        f'1_vtln_to_2_vtln_ppv McNemar\'s test p-value: {result_1_vtln_to_2_vtln_ppv.pvalue}')
    print("Cohen's Kappa kappa_1_vtln_to_2_vtln_ppv:", kappa_1_vtln_to_2_vtln_ppv)

    kappa_1_no_vtln_to_2_no_vtln_uar = cohen_kappa_score(
        conc_arr['prediction_sys_1_no_VTLN_UAR'], conc_arr['prediction_sys_2_no_VTLN_UAR'])
    table_1_no_vtln_to_2_no_vtln_uar = confusion_matrix(
        conc_arr['prediction_sys_1_no_VTLN_UAR'], conc_arr['prediction_sys_2_no_VTLN_UAR'])
    result_table_1_no_vtln_to_2_no_vtln_uar = mcnemar(
        table_1_no_vtln_to_2_no_vtln_uar, exact=True)
    print(
        f'1_no_vtln_to_2_no_vtln_uar McNemar\'s test p-value: {result_table_1_no_vtln_to_2_no_vtln_uar.pvalue}')
    print("Cohen's Kappa kappa_1_no_vtln_to_2_no_vtln_uar:",
          kappa_1_no_vtln_to_2_no_vtln_uar)

    kappa_1_no_vtln_to_2_no_vtln_ppv = cohen_kappa_score(
        conc_arr['prediction_sys_1_no_VTLN_PPV'], conc_arr['prediction_sys_2_no_VTLN_PPV'])
    table_1_no_vtln_to_2_no_vtln_ppv = confusion_matrix(
        conc_arr['prediction_sys_1_no_VTLN_PPV'], conc_arr['prediction_sys_2_no_VTLN_PPV'])
    result_table_1_no_vtln_to_2_no_vtln_ppv = mcnemar(
        table_1_no_vtln_to_2_no_vtln_ppv, exact=True)
    print(
        f'1_no_vtln_to_2_no_vtln_ppv McNemar\'s test p-value: {result_table_1_no_vtln_to_2_no_vtln_ppv.pvalue}')
    print("Cohen's Kappa kappa_1_no_vtln_to_2_no_vtln_ppv:",
          kappa_1_no_vtln_to_2_no_vtln_ppv)
############################################################################

    kappa_True_to_no_1_vtln_uar = cohen_kappa_score(
        conc_arr['True_echo'], conc_arr['prediction_sys_1_no_VTLN_UAR'])
    # Form confusion matrix for McNemar's test
    tableTrue_to_1_no_vtln_uar = confusion_matrix(
        conc_arr['True_echo'], conc_arr['prediction_sys_1_no_VTLN_UAR'])
    # Perform McNemar's test
    result_True_to_1_no_vtln_uar = mcnemar(
        tableTrue_to_1_no_vtln_uar, exact=True)
    print(
        f'True_to_no_1_vtln_uar McNemar\'s test p-value : {result_True_to_1_no_vtln_uar.pvalue}')
    print("Cohen's Kappa between kappa_True_to_no_1_vtln_uar :",
          kappa_True_to_no_1_vtln_uar)

    kappa_True_to_1_no_vtln_ppv = cohen_kappa_score(
        conc_arr['True_echo'], conc_arr['prediction_sys_1_no_VTLN_PPV'])
    # Form confusion matrix for McNemar's test
    table_True_to_1_no_vtln_ppv = confusion_matrix(
        conc_arr['True_echo'], conc_arr['prediction_sys_1_no_VTLN_PPV'])
    # Perform McNemar's test
    result_True_to_1_no_vtln_ppv = mcnemar(
        table_True_to_1_no_vtln_ppv, exact=True)
    print(
        f'True_to_1_no_vtln_ppv McNemar\'s test p-value : {result_True_to_1_no_vtln_ppv.pvalue}')
    print("Cohen's Kappa between kappa_True_to_1_no_vtln_ppv :",
          kappa_True_to_1_no_vtln_ppv)

    kappa_True_to_no_2_vtln_uar = cohen_kappa_score(
        conc_arr['True_echo'], conc_arr['prediction_sys_2_no_VTLN_UAR'])
    # Form confusion matrix for McNemar's test
    tableTrue_to_2_no_vtln_uar = confusion_matrix(
        conc_arr['True_echo'], conc_arr['prediction_sys_2_no_VTLN_UAR'])
    # Perform McNemar's test
    result_True_to_2_no_vtln_uar = mcnemar(
        tableTrue_to_2_no_vtln_uar, exact=True)
    print(
        f'True_to_no_2_vtln_uar McNemar\'s test p-value : {result_True_to_2_no_vtln_uar.pvalue}')
    print("Cohen's Kappa between kappa_True_to_no_2_vtln_uar :",
          kappa_True_to_no_2_vtln_uar)

    kappa_True_to_2_no_vtln_ppv = cohen_kappa_score(
        conc_arr['True_echo'], conc_arr['prediction_sys_2_no_VTLN_PPV'])
    # Form confusion matrix for McNemar's test
    table_True_to_2_no_vtln_ppv = confusion_matrix(
        conc_arr['True_echo'], conc_arr['prediction_sys_2_no_VTLN_PPV'])
    # Perform McNemar's test
    result_True_to_2_no_vtln_ppv = mcnemar(
        table_True_to_2_no_vtln_ppv, exact=True)
    print(
        f'True_to_2_no_vtln_ppv McNemar\'s test p-value : {result_True_to_2_no_vtln_ppv.pvalue}')
    print("Cohen's Kappa between kappa_True_to_2_no_vtln_ppv :",
          kappa_True_to_2_no_vtln_ppv)
