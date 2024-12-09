import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def predKAM(tFPA_array, weight, height, speed, alignment, bFPA):
    # normalize input features
    w = (weight - mean_weight)/std_weight
    h = (height - mean_height)/std_height
    s = (speed - mean_speed)/std_speed
    a = (alignment - mean_alignment)/std_alignment
    b = (bFPA - mean_bFPA)/std_bFPA

    rKAM_array = np.zeros((len(tFPA_array)))
    for i in range(len(tFPA_array)):
        if tFPA_array.iloc[i,0] >=13: #outside of bounds of training data
            rKAM_array[i] = 0
        else:
            t = (tFPA_array.iloc[i,0] - mean_tFPA)/std_tFPA
            pKAM = offset + b_weight*b + b_height*h + b_speed*s + b_alignment*a + b_bFPA*b + b_tFPA*t
            rKAM_array[i] = pKAM    
    return rKAM_array

# 0. SETUP
# a. set model and normalization coefficients from Rokhmanova et al. 2022
offset = 0.35673

b_weight = -0.01444
mean_weight = 71.31296
std_weight = 12.59157

b_height = -0.00413
mean_height = 172.37590
std_height = 8.63829

b_speed = 0.00340
mean_speed = 1.12072
std_speed = 0.00684

b_alignment = -0.01475
mean_alignment = 1.24025
std_alignment = 3.16196

b_bFPA = -0.00492
mean_bFPA = 3.09735
std_bFPA = 4.14900

b_tFPA = 0.30261
mean_tFPA = 5.5
std_tFPA = 2.87


# b. create arrays for storage
subs_tot = 36
store_staticAlignment = np.zeros((subs_tot,1))
store_allrKAM_NF = np.zeros((subs_tot, 80))
store_allrKAM_RT4 = np.zeros((subs_tot, 200))
store_allrKAM_RET = np.zeros((subs_tot, 200))

# 1. load meta data
root = tk.Tk()
root.withdraw() 
directory = filedialog.askdirectory()

# a. load feature array
inputFeatures_csv_file = os.path.normpath(os.path.join(directory, 'pKAM_features.csv'))
inputFeatures = pd.read_csv(inputFeatures_csv_file)
print(inputFeatures.head()) #speed, height, weight, bFPA, alignment

# b. load feedback condition ID (1:SF, 2:TF, 0:NF)
feedbackCond_csv_file = os.path.normpath(os.path.join(directory, 'feedbackGroups.csv'))
feedbackCond_file = pd.read_csv(feedbackCond_csv_file)

# 2. compute static alignment for each sub
for subject in range(1,37):
    
    print('----------------Starting analysis for subject ' + str(subject) + '--------------------')
    if feedbackCond_file.cond[subject-1] == 1:
        print('----------------CONDITION: SCALED FEEDBACK------------')
    elif feedbackCond_file.cond[subject-1] == 2:
        print('----------------CONDITION: TRINARY FEEDBACK------------')
    elif feedbackCond_file.cond[subject-1] == 0:
        print('----------------CONDITION: NO FEEDBACK------------')

    # a. open subject static file and tFPAs
    if subject < 10: 
        static_csv_file = os.path.normpath(os.path.join(directory, 's0' + str(subject)  + '\\s0' + str(subject) + '_static.csv'))
        staticDF = pd.read_csv(static_csv_file, skiprows=4)

        tFPA_NF_file = os.path.normpath(os.path.join(directory, 's0' + str(subject)  + '\\tFPA_NF.csv'))
        tFPA_NF = pd.read_csv(tFPA_NF_file)
        tFPA_RT4_file = os.path.normpath(os.path.join(directory, 's0' + str(subject)  + '\\tFPA_RT4.csv'))
        tFPA_RT4 = pd.read_csv(tFPA_RT4_file)
        tFPA_RET_file = os.path.normpath(os.path.join(directory, 's0' + str(subject)  + '\\tFPA_RET.csv'))
        tFPA_RET = pd.read_csv(tFPA_RET_file)
        

    else: 
        static_csv_file = os.path.normpath(os.path.join(directory, 's' + str(subject)  + '\\s' + str(subject) + '_static.csv'))
        staticDF = pd.read_csv(static_csv_file, skiprows=4)

        tFPA_NF_file = os.path.normpath(os.path.join(directory, 's' + str(subject)  + '\\tFPA_NF.csv'))
        tFPA_NF = pd.read_csv(tFPA_NF_file)
        tFPA_RT4_file = os.path.normpath(os.path.join(directory, 's' + str(subject)  + '\\tFPA_RT4.csv'))
        tFPA_RT4 = pd.read_csv(tFPA_RT4_file)
        tFPA_RET_file = os.path.normpath(os.path.join(directory, 's' + str(subject)  + '\\tFPA_RET.csv'))
        tFPA_RET = pd.read_csv(tFPA_RET_file)

    # b. compute right leg AJC, KJC, HJC
    MANK = np.mean(staticDF.iloc[:,11:14], axis=0)
    LANK = np.mean(staticDF.iloc[:,53:56], axis=0)
    AJCy = (MANK.iloc[1]+LANK.iloc[1])/2 # y = mediolateral
    AJCz = (MANK.iloc[2]+LANK.iloc[2])/2 # z = vertical
    
    MKNE = np.mean(staticDF.iloc[:,5:8], axis=0)
    LKNE = np.mean(staticDF.iloc[:,47:50], axis=0)
    KJCy = (MKNE.iloc[1] + LKNE.iloc[1])/2
    KJCz = (MKNE.iloc[2] + LKNE.iloc[2])/2

    # Harrington's regression equations for HJC:
    RASI = np.mean(staticDF.iloc[:,17:20], axis=0)
    LASI = np.mean(staticDF.iloc[:,14:17], axis=0)
    pelvisCentery = (RASI.iloc[1]+LASI.iloc[1])/2
    pelvisCenterz = (RASI.iloc[2]+LASI.iloc[2])/2
    ASIS_distance = np.sqrt( (RASI.iloc[1]-LASI.iloc[1])**2 + (RASI.iloc[2]-LASI.iloc[2])**2 )
    HJCy = pelvisCentery + 0.33*ASIS_distance + 7.3
    HJCz = pelvisCenterz - 0.30*ASIS_distance - 10.9

    # c. compute static knee alignment
    A = np.sqrt( (HJCy - KJCy)**2 + (HJCz - KJCz)**2 )
    B = np.sqrt( (AJCy - KJCy)**2 + (AJCz - KJCz)**2 )
    C = np.sqrt( (HJCy - AJCy)**2 + (HJCz - AJCz)**2 )

    # cosine law: 
    # TODO: check if it's 180-alignment or alignemnt-180
    alignment = 180 - np.rad2deg(np.arccos( (B**2 + A**2 - C**2)/(2*A*B) ))

    # d. store alignment; add to feature array (manually)
    store_staticAlignment[subject-1] = alignment


# 3. compute rKAM at each step
 
    # b. for all steps in NF, RT4, RET, find rKAM
    rKAM_NF = predKAM(tFPA_NF, inputFeatures.weight[subject-1], inputFeatures.height[subject-1], inputFeatures.speed[subject-1], inputFeatures.alignment[subject-1], inputFeatures.bfpa[subject-1])
    rKAM_RT4 = predKAM(tFPA_RT4, inputFeatures.weight[subject-1], inputFeatures.height[subject-1], inputFeatures.speed[subject-1], inputFeatures.alignment[subject-1], inputFeatures.bfpa[subject-1])
    rKAM_RET = predKAM(tFPA_RET, inputFeatures.weight[subject-1], inputFeatures.height[subject-1], inputFeatures.speed[subject-1], inputFeatures.alignment[subject-1], inputFeatures.bfpa[subject-1])

    # c. store subject rKAMs
    store_allrKAM_NF[subject-1] = rKAM_NF
    store_allrKAM_RT4[subject-1] = rKAM_RT4
    store_allrKAM_RET[subject-1] = rKAM_RET


# 4. visualize
SF_rows = np.where(feedbackCond_file.cond == 1)[0]
TF_rows = np.where(feedbackCond_file.cond == 2)[0]
NF_rows = np.where(feedbackCond_file.cond == 0)[0]

# ___________ FIG 1 _________________________________
# mean rKAM at RT4
fig, ax = plt.subplots(figsize = (6,6))

sf_data = np.mean(store_allrKAM_RT4[SF_rows],1) 
print(sf_data)
tf_data = np.mean(store_allrKAM_RT4[TF_rows],1) 
print(tf_data)
nf_data = np.mean(store_allrKAM_RT4[NF_rows],1) 
print(nf_data)

violin_parts = ax.violinplot([sf_data, tf_data, nf_data], 
                             positions=[1, 2, 3], 
                             showmeans=True, 
                             showextrema=True, 
                             showmedians=False)

ax.set_title('rKAM across groups, RT4')
ax.set_ylabel('rKAM')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['SF', 'TF', 'NF'])

for i, data in enumerate([sf_data, tf_data, nf_data], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(data)), data, alpha=0.3, s=15)

plt.show()

# __________ FIG 2 _____________________________
# mean rKAM at RET
fig, ax = plt.subplots(figsize = (6,6))

sf_data = np.mean(store_allrKAM_RET[SF_rows],1) 
print(sf_data)
tf_data = np.mean(store_allrKAM_RET[TF_rows],1) 
print(tf_data)
nf_data = np.mean(store_allrKAM_RET[NF_rows],1) 
print(nf_data)

violin_parts = ax.violinplot([sf_data, tf_data, nf_data], 
                             positions=[1, 2, 3], 
                             showmeans=True, 
                             showextrema=True, 
                             showmedians=False)

ax.set_title('rKAM across groups, RET')
ax.set_ylabel('rKAM')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['SF', 'TF', 'NF'])

for i, data in enumerate([sf_data, tf_data, nf_data], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(data)), data, alpha=0.3, s=15)
plt.show()

print('_______________')
print('sf mean: ' + str(np.mean(sf_data)) + ' sd: ' + str(np.std(sf_data)))
print('tf mean: ' + str(np.mean(tf_data)) + ' sd: ' + str(np.std(tf_data)))
print('nf mean: ' + str(np.mean(nf_data)) + ' sd: ' + str(np.std(nf_data)))