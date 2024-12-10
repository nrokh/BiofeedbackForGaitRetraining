import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import os

# 0. SETUP
# a. Get the desired directory to load/save the data
root = tk.Tk()
root.withdraw() 
directory = filedialog.askdirectory()

# b. load feedback condition
feedbackCond_csv_file = os.path.normpath(os.path.join(directory, 'feedbackGroups.csv'))
feedbackCond_file = pd.read_csv(feedbackCond_csv_file)

# c. load bFPA
in_bFPA_file = os.path.normpath(os.path.join(directory, 'features\\in_bFPA.csv'))
in_bFPA = np.genfromtxt(in_bFPA_file, delimiter=',')
baselineFPA = in_bFPA[1:]
targetFPA = in_bFPA[1:] - 10.0

# d. create variables to store duration
store_duration_RT1 = np.zeros((36,200))
store_duration_RT4 = np.zeros((36,200))

# 1. Compute durations

for subject in range(1,37):
    duration_RT1 = np.zeros((200,1))
    duration_RT4 = np.zeros((200,1))

    print('----------------Starting analysis for subject ' + str(subject) + '--------------------')
    if feedbackCond_file.cond[subject-1] == 1:
        print('----------------CONDITION: SCALED FEEDBACK------------')
    elif feedbackCond_file.cond[subject-1] == 2:
        print('----------------CONDITION: TRINARY FEEDBACK------------')
    elif feedbackCond_file.cond[subject-1] == 0:
        print('----------------CONDITION: NO FEEDBACK------------')

    print('target FPA: ' + str(targetFPA[subject-1]))


    # a. load toe-in 
    if subject < 10: 
        toein1_csv_file = os.path.normpath(os.path.join(directory, 's0' + str(subject)  + '\\s0' + str(subject) + '_meanFPA_1.csv'))
        tFPA_RT1 = pd.read_csv(toein1_csv_file)

        toein4_csv_file = os.path.normpath(os.path.join(directory, 's0' + str(subject)  + '\\s0' + str(subject) + '_meanFPA_4.csv'))
        tFPA_RT4 = pd.read_csv(toein4_csv_file)
        
    else: 
        toein1_csv_file = os.path.normpath(os.path.join(directory, 's' + str(subject)  + '\\s' + str(subject) + '_meanFPA_1.csv'))
        tFPA_RT1 = pd.read_csv(toein1_csv_file)

        toein4_csv_file = os.path.normpath(os.path.join(directory, 's' + str(subject)  + '\\s' + str(subject) + '_meanFPA_4.csv'))
        tFPA_RT4 = pd.read_csv(toein4_csv_file)


    # b. compute durations
    for step in range(len(tFPA_RT1)):
        if feedbackCond_file.cond[subject-1] == 1:
            if tFPA_RT1.iloc[step,2] < targetFPA[subject-1] - 2: # too far in
                duration_RT1[step] = - (abs((tFPA_RT1.iloc[step,2] - targetFPA[subject-1]))*108 - 156)
                if duration_RT1[step] < -600:
                    duration_RT1[step] = 600
                if duration_RT1[step] > -60:
                    duration_RT1[step] = 0
            elif tFPA_RT1.iloc[step,2] > targetFPA[subject-1] + 2: # too far out
                duration_RT1[step] = (abs((tFPA_RT1.iloc[step,2] - targetFPA[subject-1]))*108 - 156)
                if duration_RT1[step] > 600:
                    duration_RT1[step] = 600
                if duration_RT1[step] < 60:
                    duration_RT1[step] = 0

        if feedbackCond_file.cond[subject-1] == 2:
            if tFPA_RT1.iloc[step,2] < targetFPA[subject-1] - 2: # too far in
                duration_RT1[step] = -200

            elif tFPA_RT1.iloc[step,2] > targetFPA[subject-1] + 2: # too far out
                duration_RT1[step] = 200

    for step in range(len(tFPA_RT4)):
        if feedbackCond_file.cond[subject-1] == 1:
            if tFPA_RT1.iloc[step,2] < targetFPA[subject-1] - 2: # too far in
                duration_RT4[step] = - (abs((tFPA_RT4.iloc[step,2] - targetFPA[subject-1]))*108 - 156)
                if duration_RT4[step] < -600:
                    duration_RT4[step] = 600
                if duration_RT4[step] > -60:
                    duration_RT4[step] = 0
            elif tFPA_RT1.iloc[step,2] > targetFPA[subject-1] + 2: # too far out
                duration_RT4[step] = (abs((tFPA_RT4.iloc[step,2] - targetFPA[subject-1]))*108 - 156)
                if duration_RT4[step] > 600:
                    duration_RT4[step] = 600
                if duration_RT4[step] < 60:
                    duration_RT4[step] = 0

        if feedbackCond_file.cond[subject-1] == 2:
            if tFPA_RT4.iloc[step,2] < targetFPA[subject-1] - 2: # too far in
                duration_RT4[step] = -200

            elif tFPA_RT4.iloc[step,2] > targetFPA[subject-1] + 2: # too far out
                duration_RT4[step] = 200

    # remove zeros
    duration_RT1[duration_RT1 == 0] = np.nan
    duration_RT4[duration_RT4 == 0] = np.nan

    store_duration_RT1[subject-1] = duration_RT1[:,0]
    store_duration_RT4[subject-1] = duration_RT4[:,0]

# 3. Plot distributions
SF_rows = np.where(feedbackCond_file.cond == 1)[0]
TF_rows = np.where(feedbackCond_file.cond == 2)[0]
NF_rows = np.where(feedbackCond_file.cond == 0)[0]

fig, ax = plt.subplots(figsize = (6,6))

sns.histplot(store_duration_RT1[SF_rows].flatten(), binwidth=30, kde=True, color='#0f4c5c', alpha=0.5, label='RT1')
sns.histplot(store_duration_RT4[SF_rows].flatten(), binwidth=30, kde=True, color='#0f4c5c', alpha=0.7, label='RT4')

sns.histplot(store_duration_RT1[TF_rows].flatten(), binwidth=30, color='#5f0f40', alpha=0.5, label='RT1')
sns.histplot(store_duration_RT4[TF_rows].flatten(), binwidth=30, color='#5f0f40', alpha=0.7, label='RT4')

plt.show()

