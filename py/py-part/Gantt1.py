# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:40:16 2023

@author: 97254
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

if __name__ == '__main__':
    df = pd.read_excel(r"C:\BGU\Project\Research Proposal\Gantt.xlsx")
    df = df.sort_values('End', ascending=False)  # sort by end date in reverse order
    
    # project start date
    proj_start = df.Start.min()
    # number of days from project start to task start
    df['start_num'] = (df.Start-proj_start).dt.days
    # number of days from project start to end of tasks
    df['end_num'] = (df.End-proj_start).dt.days
    # days between start and end of each task
    df['days_start_to_end'] = df.end_num - df.start_num
    
    c_dict = {'Literature review':'#E64646', 'Database':'#E69646', 'VTLN':'#34D05C',
              'Echolalia detection':'#34D0C3', 'Echolalia score':'#3475D0',
              'Presentation':'#fcc603','Results analysis':'#4a4882',
              'Poster':'#2bff00','Preparing final report':'#f00e92'}

    # create a column with the color for each department
    def color(row):
        c_dict = {'Literature review':'#E64646', 'Database':'#E69646', 'VTLN':'#34D05C',
                  'Echolalia detection':'#34D0C3', 'Echolalia score':'#3475D0',
                  'Presentation':'#fcc603','Results analysis':'#4a4882',
                  'Poster':'#22d661','Preparing final report':'#f00e92'}

        return c_dict[row['Task']]
    
    
    df['color'] = df.apply(color, axis=1)
    # days between start and current progression of each task
    df['current_num'] = (df.days_start_to_end * df.Completion)
    
    
    
    
    fig, ax = plt.subplots(1, figsize=(16,6))
    # bars
    ax.barh(df.Task, df.current_num, left=df.start_num, color=df.color)
    ax.barh(df.Task, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.5)
    # texts
    for idx, row in df.iterrows():
        ax.text(row.end_num+0.1, idx, 
                f"{int(row.Completion*100)}%", 
                va='center_baseline', alpha=0.8)
    ##### LEGENDS #####
   
    legend_elements = [Patch(facecolor=c_dict[i], label=i)  for i in c_dict]
    plt.legend(handles=legend_elements,fontsize=14)
    ##### TICKS #####
    xticks = np.arange(0, df.end_num.max()+1, 20)
    xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
    xticks_minor = np.arange(0, df.end_num.max()+1, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels[::20])
    plt.show()