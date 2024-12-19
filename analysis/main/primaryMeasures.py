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

def calculate_responsiveness(input_FPA, targetFPA):
    incorrect_steps = 0
    resp_tally = 0
    
    for step in range(len(input_FPA) - 1):
        current_fpa = input_FPA.iloc[step, 2]
        next_fpa = input_FPA.iloc[step + 1, 2]
        
        if current_fpa < targetFPA - 2:
            incorrect_steps += 1
            if next_fpa > current_fpa:
                resp_tally += 1
        
        elif current_fpa > targetFPA + 2:
            incorrect_steps += 1
            if next_fpa < current_fpa:
                resp_tally += 1
    
    store_resp = resp_tally / incorrect_steps if incorrect_steps > 0 else 0
    
    return store_resp

# 0. Create general vars
subs_tot = 36
# Accuracy
store_inRangePercent = np.zeros((subs_tot, 6)) 
store_inRangeNFPercent = np.zeros((subs_tot, 6)) 
# RMSE
store_RMSE = np.zeros((subs_tot, 6))
store_C_RMSE = np.zeros((subs_tot, 6))
# Responsiveness
store_resp = np.zeros((subs_tot, 6))

# 1. Load data
# a. Get the desired directory 
root = tk.Tk()
root.withdraw() 
directory = filedialog.askdirectory()

# b. Load feedback condition ID (1:SF, 2:TF, 0:NF)
feedbackCond_csv_file = os.path.normpath(os.path.join(directory, 'main\\feedbackGroups.csv'))
feedbackCond_file = pd.read_csv(feedbackCond_csv_file)

# c. Iterate through subjects 
for subject in range(1,37):
    print('----------------Starting analysis for subject ' + str(subject) + '--------------------')
    if feedbackCond_file.cond[subject-1] == 1:
        print('----------------CONDITION: SCALED FEEDBACK------------')
    elif feedbackCond_file.cond[subject-1] == 2:
        print('----------------CONDITION: TRINARY FEEDBACK------------')
    elif feedbackCond_file.cond[subject-1] == 0:
        print('----------------CONDITION: NO FEEDBACK------------')
    else:
        print('!!!--------ERROR: check feedback condition file?---------!!!')

    if subject < 10:
        # Load FPAs:
        baseline_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_baseline_meanFPA.csv'))
        baselineFPA = pd.read_csv(baseline_csv_file)

        nf_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_noFB_meanFPA.csv'))
        nfFPA = pd.read_csv(nf_csv_file)

        toein1_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_meanFPA_1.csv'))
        toein1FPA = pd.read_csv(toein1_csv_file)

        toein2_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_meanFPA_2.csv'))
        toein2FPA = pd.read_csv(toein2_csv_file)

        toein3_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_meanFPA_3.csv'))
        toein3FPA = pd.read_csv(toein3_csv_file)

        toein4_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_meanFPA_4.csv'))
        toein4FPA = pd.read_csv(toein4_csv_file)

        ret_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s0' + str(subject)  + '\\s0' + str(subject) + '_retention_meanFPA.csv'))
        retFPA = pd.read_csv(ret_csv_file)

        fullFPA = pd.concat([baselineFPA, toein1FPA, toein2FPA, toein3FPA, toein4FPA, retFPA])
    else:
        baseline_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_baseline_meanFPA.csv'))
        baselineFPA = pd.read_csv(baseline_csv_file)

        nf_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_noFB_meanFPA.csv'))
        nfFPA = pd.read_csv(nf_csv_file)

        toein1_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_meanFPA_1.csv'))
        toein1FPA = pd.read_csv(toein1_csv_file)

        toein2_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_meanFPA_2.csv'))
        toein2FPA = pd.read_csv(toein2_csv_file)

        toein3_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_meanFPA_3.csv'))
        toein3FPA = pd.read_csv(toein3_csv_file)

        toein4_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_meanFPA_4.csv'))
        toein4FPA = pd.read_csv(toein4_csv_file)

        ret_csv_file = os.path.normpath(os.path.join(directory, 'processedData\\s' + str(subject)  + '\\s' + str(subject) + '_retention_meanFPA.csv'))
        retFPA = pd.read_csv(ret_csv_file)

        fullFPA = pd.concat([baselineFPA, toein1FPA, toein2FPA, toein3FPA, toein4FPA, retFPA])

    # d. Compute baseline FPA
    bFPA_deg = np.mean(baselineFPA.iloc[:,2])
    print('Baseline FPA was: ' + str(bFPA_deg) + '(' + str(np.std(baselineFPA.iloc[:,2])) + ')')

    # e. Compute accuracy
    targetFPA = bFPA_deg-10

    # i. nominal condition
    inRange_nf = 100*len(nfFPA[(nfFPA.iloc[:,2]<targetFPA+2) & (nfFPA.iloc[:,2]>targetFPA-2)])/len(nfFPA)
    print('percent steps in range during NF: ' + str(inRange_nf))

    # ii. toe-in trials
    inRange_t1 = 100 * len(pd.concat([toein1FPA.iloc[:80], toein1FPA.iloc[121:]])[(pd.concat([toein1FPA.iloc[:80], toein1FPA.iloc[121:]]).iloc[:, 2] < targetFPA + 2) & (pd.concat([toein1FPA.iloc[:80], toein1FPA.iloc[121:]]).iloc[:, 2] > targetFPA - 2)]) / (len(toein1FPA) - 41)
    print('percent steps in range during toe-in 1: ' + str(inRange_t1))
    inRange_t2 = 100 * len(pd.concat([toein2FPA.iloc[:80], toein2FPA.iloc[121:]])[(pd.concat([toein2FPA.iloc[:80], toein2FPA.iloc[121:]]).iloc[:, 2] < targetFPA + 2) & (pd.concat([toein2FPA.iloc[:80], toein2FPA.iloc[121:]]).iloc[:, 2] > targetFPA - 2)]) / (len(toein2FPA) - 41)
    print('percent steps in range during toe-in 2: ' + str(inRange_t2))
    inRange_t3 = 100 * len(pd.concat([toein3FPA.iloc[:80], toein3FPA.iloc[121:]])[(pd.concat([toein3FPA.iloc[:80], toein3FPA.iloc[121:]]).iloc[:, 2] < targetFPA + 2) & (pd.concat([toein3FPA.iloc[:80], toein3FPA.iloc[121:]]).iloc[:, 2] > targetFPA - 2)]) / (len(toein3FPA) - 41)
    print('percent steps in range during toe-in 3: ' + str(inRange_t3))
    inRange_t4 = 100 * len(pd.concat([toein4FPA.iloc[:80], toein4FPA.iloc[121:]])[(pd.concat([toein4FPA.iloc[:80], toein4FPA.iloc[121:]]).iloc[:, 2] < targetFPA + 2) & (pd.concat([toein4FPA.iloc[:80], toein4FPA.iloc[121:]]).iloc[:, 2] > targetFPA - 2)]) / (len(toein4FPA) - 41)
    print('percent steps in range during toe-in 4: ' + str(inRange_t4))

    # iii. catch trials
    inRange_c1 = 100 * len(toein1FPA.iloc[80:121][(toein1FPA.iloc[80:121, 2] < targetFPA + 2) & (toein1FPA.iloc[80:121, 2] > targetFPA - 2)]) / 40
    print('percent steps in range during catch 1: ' + str(inRange_c1)) 
    inRange_c2 = 100 * len(toein2FPA.iloc[80:121][(toein2FPA.iloc[80:121, 2] < targetFPA + 2) & (toein2FPA.iloc[80:121, 2] > targetFPA - 2)]) / 40
    print('percent steps in range during catch 2: ' + str(inRange_c2))
    inRange_c3 = 100 * len(toein3FPA.iloc[80:121][(toein3FPA.iloc[80:121, 2] < targetFPA + 2) & (toein3FPA.iloc[80:121, 2] > targetFPA - 2)]) / 40
    print('percent steps in range during catch 3: ' + str(inRange_c3))
    inRange_c4 = 100 * len(toein4FPA.iloc[80:121][(toein4FPA.iloc[80:121, 2] < targetFPA + 2) & (toein4FPA.iloc[80:121, 2] > targetFPA - 2)]) / 40
    print('percent steps in range during catch 4: ' + str(inRange_c4))

    # iv. retention trial
    inRange_ret = 100*len(retFPA[(retFPA.iloc[:,2]<targetFPA+2) & (retFPA.iloc[:,2]>targetFPA-2)])/len(retFPA)
    print('percent steps in range during retention: ' + str(inRange_ret)) 

    # v. store all
    inRange_all = [inRange_nf, inRange_t1, inRange_t2, inRange_t3, inRange_t4, inRange_ret]
    inRange_NFs = [inRange_nf, inRange_c1, inRange_c2, inRange_c3, inRange_c4, inRange_ret]
    store_inRangePercent[subject-1] = inRange_all
    store_inRangeNFPercent[subject-1] = inRange_NFs

    # f. compute RMSE:
    # i. nominal
    RMSENF = np.sqrt(np.mean( (bFPA_deg-10-nfFPA.iloc[:,2])**2 ))
    
    # ii. toe-in trials
    RMSET1 = np.sqrt(np.mean( (bFPA_deg-10-toein1FPA.iloc[:,2])**2 ))
    RMSET2 = np.sqrt(np.mean( (bFPA_deg-10-toein2FPA.iloc[:,2])**2 ))
    RMSET3 = np.sqrt(np.mean( (bFPA_deg-10-toein3FPA.iloc[:,2])**2 ))
    RMSET4 = np.sqrt(np.mean( (bFPA_deg-10-toein4FPA.iloc[:,2])**2 ))
    
    # iii. catch trials
    RMSEC1 = np.sqrt(np.mean( (bFPA_deg-10-toein1FPA.iloc[80:121,2])**2 ))
    RMSEC2 = np.sqrt(np.mean( (bFPA_deg-10-toein2FPA.iloc[80:121,2])**2 ))
    RMSEC3 = np.sqrt(np.mean( (bFPA_deg-10-toein3FPA.iloc[80:121,2])**2 ))
    RMSEC4 = np.sqrt(np.mean( (bFPA_deg-10-toein4FPA.iloc[80:121,2])**2 ))
    
    # iv. retention
    RMSER = np.sqrt(np.mean( (bFPA_deg-10-retFPA.iloc[:,2])**2 ))

    # v. store all    
    RMSE_all = [RMSENF, RMSET1, RMSET2, RMSET3, RMSET4, RMSER]
    store_RMSE[subject-1] = RMSE_all

    C_RMSE_all = [RMSENF, RMSEC1, RMSEC2, RMSEC3, RMSEC4, RMSER]
    store_C_RMSE[subject-1] = C_RMSE_all

    # g. compute responsiveness
    store_resp_NF = calculate_responsiveness(nfFPA, targetFPA)
    store_resp_RT1 = calculate_responsiveness(toein1FPA, targetFPA)
    store_resp_RT2 = calculate_responsiveness(toein2FPA, targetFPA)
    store_resp_RT3 = calculate_responsiveness(toein3FPA, targetFPA)
    store_resp_RT4 = calculate_responsiveness(toein4FPA, targetFPA)
    store_resp_RET = calculate_responsiveness(retFPA, targetFPA)

    resp_all = [store_resp_NF, store_resp_RT1, store_resp_RT2, store_resp_RT3, store_resp_RT4, store_resp_RET]
    store_resp[subject-1] = resp_all

# 2. Plot group means and cumulative results
SF_rows = np.where(feedbackCond_file.cond == 1)[0]
TF_rows = np.where(feedbackCond_file.cond == 2)[0]
NF_rows = np.where(feedbackCond_file.cond == 0)[0]

# i. Accuracy over time
x = np.arange(6)
plt.figure(figsize=(6,6))
plt.title('Accuracy over time')
plt.plot(x-0.05, np.mean(store_inRangePercent[SF_rows], axis=0), '-o', color = '#0f4c5c', label = 'SF')
plt.errorbar(x-0.05, np.mean(store_inRangePercent[SF_rows], axis=0), yerr=np.std(store_inRangePercent[SF_rows], axis=0), fmt='none', ecolor='#0f4c5c', capsize=5)

plt.plot(x, np.mean(store_inRangePercent[TF_rows], axis=0), '-o', color = '#5f0f40', label = 'TF')
plt.errorbar(x, np.mean(store_inRangePercent[TF_rows], axis=0), yerr=np.std(store_inRangePercent[TF_rows], axis=0), fmt='none', ecolor='#5f0f40', capsize=5)

plt.plot(x+0.05, np.mean(store_inRangePercent[NF_rows], axis=0), '-o', color = '#e36414', label = 'NF')
plt.errorbar(x+0.05, np.mean(store_inRangePercent[NF_rows], axis=0), yerr=np.std(store_inRangePercent[NF_rows], axis=0), fmt='none', ecolor='#e36414', capsize=5)

plt.legend()
plt.ylim([0,100])
plt.ylabel('Steps within target range (%)')
plt.show()

# ii. RMSE over time
plt.figure(figsize=(6,6))
plt.title('RMSE over time')
plt.plot(x-0.05, np.mean(store_RMSE[SF_rows], axis=0), '-o', color = '#0f4c5c', label = 'SF')
plt.errorbar(x-0.05, np.mean(store_RMSE[SF_rows], axis=0), yerr=np.std(store_RMSE[SF_rows], axis=0), fmt='none', ecolor='#0f4c5c', capsize=5)

plt.plot(x, np.mean(store_RMSE[TF_rows], axis=0), '-o', color = '#5f0f40', label = 'TF')
plt.errorbar(x, np.mean(store_RMSE[TF_rows], axis=0), yerr=np.std(store_RMSE[TF_rows], axis=0), fmt='none', ecolor='#5f0f40', capsize=5)

plt.plot(x+0.05, np.mean(store_RMSE[NF_rows], axis=0), '-o', color = '#e36414', label = 'NF')
plt.errorbar(x+0.05, np.mean(store_RMSE[NF_rows], axis=0), yerr=np.std(store_RMSE[NF_rows], axis=0), fmt='none', ecolor='#e36414', capsize=5)

plt.legend()
plt.ylim([0,12])
plt.ylabel('RMSE (deg)')
plt.show()

# iii. RMSE over time (catch trials)
plt.figure(figsize=(6,6))
plt.title('catch trial RMSEs')
plt.plot(x-0.05, np.mean(store_C_RMSE[SF_rows], axis=0), '-o', color = '#0f4c5c', label = 'SF')
plt.errorbar(x-0.05, np.mean(store_C_RMSE[SF_rows], axis=0), yerr=np.std(store_C_RMSE[SF_rows], axis=0), fmt='none', ecolor='#0f4c5c', capsize=5)

plt.plot(x, np.mean(store_C_RMSE[TF_rows], axis=0), '-o', color = '#5f0f40', label = 'TF')
plt.errorbar(x, np.mean(store_C_RMSE[TF_rows], axis=0), yerr=np.std(store_C_RMSE[TF_rows], axis=0), fmt='none', ecolor='#5f0f40', capsize=5)

plt.plot(x+0.05, np.mean(store_C_RMSE[NF_rows], axis=0), '-o', color = '#e36414', label = 'NF')
plt.errorbar(x+0.05, np.mean(store_C_RMSE[NF_rows], axis=0), yerr=np.std(store_C_RMSE[NF_rows], axis=0), fmt='none', ecolor='#e36414', capsize=5)
plt.show()

# iv. change in RMSE (NF vs. RT4)
fig, ax = plt.subplots(figsize = (6,6))
sf_data = 100*(store_RMSE[SF_rows,4] - store_RMSE[SF_rows,0])/store_RMSE[SF_rows,0]
tf_data = 100*(store_RMSE[TF_rows,4] - store_RMSE[TF_rows,0])/store_RMSE[TF_rows,0]
nf_data = 100*(store_RMSE[NF_rows,4] - store_RMSE[NF_rows,0])/store_RMSE[NF_rows,0]
violin_parts = ax.violinplot([sf_data, tf_data, nf_data], 
                             positions=[1, 2, 3], 
                             showmeans=True, 
                             showextrema=True, 
                             showmedians=False)
ax.set_title('Change in RMSE between NF and RT4')
ax.set_ylabel('Relative change in RMSE')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['SF', 'TF', 'NF'])

for i, data in enumerate([sf_data, tf_data, nf_data], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(data)), data, alpha=0.3, s=15)
plt.ylim([-85, 150])
plt.show()

# v. change in RMSE (NF vs RET)
fig, ax = plt.subplots(figsize = (6,6))
sf_data = 100*(store_RMSE[SF_rows,5] - store_RMSE[SF_rows,0])/store_RMSE[SF_rows,0]
tf_data = 100*(store_RMSE[TF_rows,5] - store_RMSE[TF_rows,0])/store_RMSE[TF_rows,0]
nf_data = 100*(store_RMSE[NF_rows,5] - store_RMSE[NF_rows,0])/store_RMSE[NF_rows,0]
violin_parts = ax.violinplot([sf_data, tf_data, nf_data], 
                             positions=[1, 2, 3], 
                             showmeans=True, 
                             showextrema=True, 
                             showmedians=False)
ax.set_title('Change in RMSE between NF and Retention')
ax.set_ylabel('Relative change in RMSE')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['SF', 'TF', 'NF'])

for i, data in enumerate([sf_data, tf_data, nf_data], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(data)), data, alpha=0.3, s=15)
plt.show()

# vi. change in RMSE (catch trial)
fig, ax = plt.subplots(figsize = (6,6))
sf_data = 100*(store_C_RMSE[SF_rows,4] - store_RMSE[SF_rows,0])/store_RMSE[SF_rows,0]
tf_data = 100*(store_C_RMSE[TF_rows,4] - store_RMSE[TF_rows,0])/store_RMSE[TF_rows,0]
nf_data = 100*(store_C_RMSE[NF_rows,4] - store_RMSE[NF_rows,0])/store_RMSE[NF_rows,0]
violin_parts = ax.violinplot([sf_data, tf_data, nf_data], 
                             positions=[1, 2, 3], 
                             showmeans=True, 
                             showextrema=True, 
                             showmedians=False)
ax.set_title('Change in RMSE between NF and catch RT4')
ax.set_ylabel('Relative change in RMSE')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['SF', 'TF', 'NF'])

for i, data in enumerate([sf_data, tf_data, nf_data], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(data)), data, alpha=0.3, s=15)
plt.show()
