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

# 0. Setup
root = tk.Tk()
root.withdraw() 
directory = filedialog.askdirectory()

# a. select the file that has all the subject folders
feedbackCond_csv_file = os.path.normpath(os.path.join(directory, 'feedbackGroups.csv'))
fb_cond = pd.read_csv(feedbackCond_csv_file).values.flatten()[:36] # 2 = SF, 1 = TF, 0 = NF

# b. open files
files = ['in_proprio_in.csv', 'in_proprio_out.csv']
data = {}
for file in files:
    file_path = os.path.normpath(os.path.join(directory, 'features', file))
    data[file.split('.')[0]] = np.abs(np.genfromtxt(file_path, delimiter=',')) if 'RMSE' not in file else np.genfromtxt(file_path, delimiter=',')

in_proprio_in = -data['in_proprio_in'][1:].flatten()
in_proprio_out = data['in_proprio_out'][1:].flatten()

# 1. Process
# a. Make a dataframe for the proprioception data with the feedback condition
df = pd.DataFrame({'Feedback Condition': fb_cond, 'Toe-In': in_proprio_in, 'Toe-Out': in_proprio_out})

# b. Plot the toe-in and toe-out proprioception data 
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# c. Define a function to plot the data
def plot_proprioception_error(ax, data, title):
    feedback_conditions = ['SF', 'TF', 'NF']
    violin_parts = ax.violinplot(data, positions=range(1, 4), showmeans=True, showextrema=True, showmedians=False)
    ax.set_title(title)
    ax.set_ylabel('Proprioceptive Error (degrees)')
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels(feedback_conditions)
    for i, cond_data in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, 0.04, len(cond_data)), cond_data, alpha=0.3, s=15)

# d. Separate data for each feedback condition
data_in = [df[df['Feedback Condition'] == i]['Toe-In'] for i in range(2, -1, -1)]
data_out = [df[df['Feedback Condition'] == i]['Toe-Out'] for i in range(2, -1, -1)]

# 2. Plot Proprioceptive Error
plot_proprioception_error(axs[0], data_in, 'Toe-In Proprioceptive Error by Feedback Condition')
plot_proprioception_error(axs[1], data_out, 'Toe-Out Proprioceptive Error by Feedback Condition')

plt.tight_layout()
plt.show()

print(np.mean(data_in, axis = 1))
print(np.std(data_in, axis=1))
print(np.mean(data_out, axis = 1))
print(np.std(data_out, axis=1))
