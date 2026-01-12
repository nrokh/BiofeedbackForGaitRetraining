# Does Vibrotactile Biofeedback Improve Gait Retraining Performance? 

This repository contains all the necessary code to reproduce the experimental design and analytical findings described in our manuscript.  

**Citation:**  
Rokhmanova N, Sundaram VH, Halilaj E, Kuchenbecker KJ (_under review_) Does Vibrotactile Biofeedback Improve Gait Retraining Performance? 

## Data availability
The full set of raw data are available in the Edmond repository linked [here](https://edmond.mpg.de/privateurl.xhtml?token=6656b649-fb6c-4941-91cc-e57072fb158f). The code shared in this repository outlines the full data processing pathway, from raw data to processed features.

In addition, all post-processed features needed to run the statistical analyses and visualize experimental results are also shared here for convenience and ease in replicability. 

# Getting started
This repository includes MATLAB and Python code. 

## Running the experiment
### Imports
Real-time streaming of marker data into Python relies on the Vicon DataStream SDK, available [here](https://www.vicon.com/software/datastream-sdk/). Ensure that you have installed it correctly and import it as:  
```
from vicon_dssdk import ViconDataStream
```

After processing gait events and kinematics, vibration triggers are sent to the ARIADNE device, an open-source wearable system for providing vibrotactile motion guidance; details on building and interacting with ARIADNE can be found [here](https://github.com/nrokh/ARIADNE). Real-time interaction with ARIADNE requires:
```
import asyncio
from bleak import BleakScanner, BleakClient
```

Additionally, you will want to include the following standard packages:
```
import argparse
import sys
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keyboard
import struct
import tkinter as tk
from tkinter import filedialog
import os
```
### What's included
* **01_acc_measure.mlapp**   
    * This MATLAB application measures the skin's response to the ARIADNE vibrotactors mounted on the shank. These data can be helpful in ensuring that vibrations are of a similar amplitude across the medial and lateral sides, on subsequent days of the experiment, and across different participants.   
* **02_vbtest.mlapp**  
    * This MATLAB application is used to evaluate a subject's accuracy in discriminating ARIADNE vibrations of varying durations. 
* **03_propriotest.py**  
    * This Python script is used to evaluate a subject's proprioceptive sensitivity in their ankle angle (toe-in and toe-out).
* **04_ROM.py**  
    * This Python script is used to evaluate a subject's dynamic range of motion (ROM) in toe-in and toe-out while they walk on a laboratory treadmill.
* **05_baseline.py**  
    * This Python script is used to compute a subject's baseline foot progression angle while they walk for one minute on a treadmill at a cadence of 80 steps per minute.
* **06_NF.py**  
    * This Python script is used to assess a subject's nominal ability to maintain a 10-degree toe-in foot progression angle, relative to the baseline angle computed in **05_baseline.py**, before any feedback or training has commenced.
* **07_TI.py**  
    * This Python script is used for a 5-minute block of gait retraining. Subjects are assigned to one of three groups: _No Feedback_, where they continue trying to walk with a 10-degree toe-in angle as they did in **06_NF.py**; _Trinary Feedback_, where subjects receive vibrations 330ms in duration to the medial or lateral ARIADNE actuator, once per step, based on how close the step was to their target 10-degree toe-in angle; and _Scaled Feedback_, where the duration of vibration scales linearly based on how close the step was to the target angle, saturating at 600ms. Halfway through the 5-minute trial, feedback is turned off for 40 steps.
* **08_retention.py**  
    * This Python script is a 5-minute block of walking with no feedback provided, used to assess how well the subjects have learned the toe-in gait after 4 blocks of **07_TI.py**.
* **09_comparison.py**  
     * This Python script is a comparison trial, for subjects to evaluate the other feedback types evaluated in the study for at least one minute each. Note that here, the target angle is 10 degrees _toe-out_ relative to their baseline foot progression angle angle. 
    

## Analyzing experimental results:
When running these scripts, make sure you have navigated to the _analysis_ folder, where you will find the following:
* **main**
   * All processing and analysis code to evaluate the primary outcome measures of this study.  
* **CCA**
   * Canonical Correlation Analysis input and output features, as well as the code to implement the analysis.
* **deltaKAM**
   * Features and scaling parameters to implement our previously-published model to predict KAM reduction magnitude.
* **proprioAndVB**
   * Visualizations of proprioceptive and vibrotactile task performance.
* **vibrationDurations**
   * Visualization comparing average vibration durations across the two feedback groups.



