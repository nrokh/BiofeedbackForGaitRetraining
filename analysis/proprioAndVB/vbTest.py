import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from scipy import stats
from matplotlib.colors import to_rgba
import ptitprince as pt 
import seaborn as sns

root = tk.Tk()
root.withdraw() 
directory = filedialog.askdirectory()

feedbackCond_csv_file = os.path.normpath(os.path.join(directory, 'proprioAndVB\\feedbackGroups.csv'))
fb_cond = pd.read_csv(feedbackCond_csv_file).values.flatten()[:36] # 2 = SF, 1 = TF, 0 = NF

subs_tot = 36


# Load VB test data from day 1, look at accuracy for each duration difference (30, 80, 240, 300 ms)
vbtest = []
duration_diff_ms = []
vibration_side = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 
                  0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0] # 1 = right, 0 = left
duration_pairs_ms = [[300 , 380], [300, 270], [330, 300], [300, 600], [270, 300], [60, 300], [220, 300], [300, 330], [380, 300],
                [300, 270], [300, 220], [300, 60], [330, 300], [300, 60], [300, 380], [220, 300], [300, 600], [270, 300], [600, 300],
                [270, 300], [60, 300], [300, 380], [300, 220], [300, 330], [300, 60], [600, 300], [300, 220], [380, 300], [330, 300],
                [600, 300], [300, 380], [300, 270], [330, 300], [300, 600], [270, 300], [60, 300], [220, 300], [300, 330], [380, 300],
                [300, 270], [300, 220], [300, 60], [330, 300], [300, 60], [300, 380], [220, 300], [300, 600], [270, 300], [600, 300],
                [270, 300], [60, 300], [300, 380], [300, 220], [300, 330], [300, 60], [600, 300], [300, 220], [380, 300], [330, 300], [600, 300]] # vibration pairs used in the VB test (ms)


sorted_duration_pairs_ms = np.array([[300, pair[1]] if pair[0] == 300 else [300, pair[0]] for pair in duration_pairs_ms])
duration_to_index = {60: 0, 220: 1, 270: 2, 330: 3, 380: 4, 600: 5}
vib_pairs = [[] for _ in range(6)]

for i, pair in enumerate(sorted_duration_pairs_ms):
    index = duration_to_index.get(pair[1])
    if index is not None:
        vib_pairs[index].append(i)

vbtest = []
for i in range(subs_tot):
    sub = i+1
    sub_str = str(sub).zfill(2)
    vbtest_file = os.path.normpath(os.path.join(directory, 'processedData\\s'+sub_str,'s'+sub_str+'_day1_vbtest.csv'))
    vbtest.append(np.genfromtxt(vbtest_file, delimiter=','))
    if sub == 4:
        vbtest[3][0] = 1

# Calculate the accuracy for each vibration pair on each side
vbtest_acc_left = np.zeros((subs_tot, 6)) # 0 = left
vbtest_acc_right = np.zeros((subs_tot, 6)) # 1 = right

for i in range(subs_tot):
    for j in range(6):
        idx = vib_pairs[j]
        for k in idx: 
            if vibration_side[k] == 0:
                if vbtest[i][k] == 1:
                    vbtest_acc_left[i][j] += 1
            else:
                if vbtest[i][k] == 1:
                    vbtest_acc_right[i][j] += 1

trial_counts_side = np.array([len(vib_pairs[j])/2 for j in range(6)])
vbtest_acc_left = vbtest_acc_left / trial_counts_side
vbtest_acc_right = vbtest_acc_right / trial_counts_side
df_left = pd.DataFrame(vbtest_acc_left, columns=[60, 220, 270, 330, 380, 600])
df_right = pd.DataFrame(vbtest_acc_right, columns=[60, 220, 270, 330, 380, 600])
df_left['Feedback Condition'] = fb_cond
df_right['Feedback Condition'] = fb_cond
df_left = pd.melt(df_left, id_vars=['Feedback Condition'], value_vars=[60, 220, 270, 330, 380, 600],
                var_name='Vibration Pair (ms)', value_name='Accuracy')
df_right = pd.melt(df_right, id_vars=['Feedback Condition'], value_vars=[60, 220, 270, 330, 380, 600],
                var_name='Vibration Pair (ms)', value_name='Accuracy')

fig, axs = plt.subplots(2,3)
feedback_conditions = ['NF', 'TF', 'SF']
min_pt_size = 60  # Minimum circle size
durations = [60, 220, 270, 330, 380, 600]

# Left side
for i, cond in enumerate(feedback_conditions):
    ax = axs[0, i]
    for duration in durations:
        subset = df_left[(df_left['Feedback Condition'] == i) & (df_left['Vibration Pair (ms)'] == duration)]
        sizes = subset.groupby('Accuracy').size().reindex(subset['Accuracy']).fillna(0).apply(lambda x: min_pt_size * x)  # Scale factor for size
        sns.scatterplot(x=[duration] * len(subset), y='Accuracy', data=subset, 
                        color='black', marker='o', s=sizes, ax=ax, label=duration)
    ax.set_title(f'VB Test Accuracy for {cond} Feedback Condition (Left Side)')
    ax.set_ylim(0, 1.1)  # Set y-axis limits
    ax.set_xticks(durations)  # Set x-axis ticks to the specific durations
    ax.axvline(x=300, color='gray', linestyle='--')  # Draw dashed vertical line at 300 ms
    ax.legend().set_visible(False)  # Turn off the legend

# Right side
for i, cond in enumerate(feedback_conditions):
    ax = axs[1, i]
    for duration in durations:
        subset = df_right[(df_right['Feedback Condition'] == i) & (df_right['Vibration Pair (ms)'] == duration)]
        sizes = subset.groupby('Accuracy').size().reindex(subset['Accuracy']).fillna(0).apply(lambda x: min_pt_size * x)  # Scale factor for size
        sns.scatterplot(x=[duration] * len(subset), y='Accuracy', data=subset, 
                        color='black', marker='o', s=sizes, ax=ax, label=duration)
    ax.set_title(f'VB Test Accuracy for {cond} Feedback Condition (Right Side)')
    ax.set_ylim(0, 1.1)  # Set y-axis limits
    ax.set_xticks(durations)  # Set x-axis ticks to the specific durations
    ax.axvline(x=300, color='gray', linestyle='--')  # Draw dashed vertical line at 300 ms
    ax.legend().set_visible(False)  # Turn off the legend

# Create a legend for the sizes of the dots
handles, labels = [], []
for trials in range(1, 13):  # 1 to 12 trials
    size = min_pt_size * trials  # Scale factor for size based on group size
    handles.append(plt.scatter([], [], s=size, color='black', alpha=0.6))
    labels.append(f'{trials} trials')

fig.legend(handles, labels, title='Dot Size (Number of Trials)', loc='upper right')
plt.tight_layout()
plt.show()

