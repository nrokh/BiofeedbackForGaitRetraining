from vicon_dssdk import ViconDataStream
import argparse
import sys
import time
import math
import pandas as pd
import numpy as np
import time
import keyboard
import tkinter as tk
from tkinter import filedialog
import os

############### VICON SETUP ########################
# create arg to host (Vicon Nexus)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default = "localhost:801")
args = parser.parse_args()

client = ViconDataStream.Client()

try:
    # Connect to Nexus (Nexus needs to be on and either Live or replaying previously collected data)
    client.Connect( args.host)
    print( '        Connected to Nexus')

    # Enable the data type
    client.EnableMarkerData()

    # Report whether data type was enabled successfully:
    print ( '        Markers enabled? ', client.IsMarkerDataEnabled() )

    # start getting frames 
    HasFrame = False
    timeout = 50
    while not HasFrame:
        print( '.' )
        try:
            if client.GetFrame():
                HasFrame = True
            timeout=timeout-1
            if timeout < 0:
                print('Failed to get frame')
                sys.exit()
        except ViconDataStream.DataStreamException as e:
            client.GetFrame()

    # Set streaming mode to Server Push (lowest latency, but buffers could overfill, resulting in dropped frames)
    client.SetStreamMode( ViconDataStream.Client.StreamMode.EServerPush)
    print( '        Current frame rate: ', client.GetFrameRate() )

    # Get the subject's name
    subjectNames = client.GetSubjectNames()
    print('        Subject name: ', subjectNames[0])
    subjectName = subjectNames[0]  # select the main subject

    # Check on markers
    client.GetFrame()  # get the frame
    marker_names = client.GetMarkerNames(subjectName)
    marker_names = [x[0] for x in marker_names]
    if 'RTOE' not in marker_names or 'RHEE' not in marker_names or 'RANK' not in marker_names:
        print("Missing markers or marker name, please check the VICON software")
        sys.exit()

    # Get the desired directory to save the data
    root = tk.Tk()
    root.withdraw() 
    directory = filedialog.askdirectory()
    csv_file_main = os.path.join(directory, subjectNames[0] + '_proprioception.csv')
    csv_file_baseAngle = os.path.join(directory, subjectNames[0] + '_baseAngles_proprioception.csv')
    counter = 0
    # Check if the file already exists
    while os.path.exists(csv_file_main):
        counter += 1
        csv_file_main = os.path.join(directory, subjectNames[0] + '_proprioception_' + str(counter) + '.csv')
        csv_file_baseAngle = os.path.join(directory, subjectNames[0] + '_baseAngles_proprioception_'+ str(counter) + '.csv')
    print('        Main data will be saved to: ', csv_file_main)

    ################# Proprioception Test ###################
    FPA_store = []
    base_angle_store = []
    err_prop = []
    err_prop_in = []
    err_prop_out = []
    deg_test = [-5, 10, -10, -15, 5, 15, 15, -15, 10, -5, 5, -10, 10, -5, 15, -15, 5, -10] #pseudo-randomized order
    for deg_i in range(len(deg_test)):

        ################# Manually moving the participant's foot to the desired angle #################
        print("        Press space after moving the participant's foot to " + str(deg_test[deg_i]) + " deg: ")
        keyboard.wait('space')  
    
        client.GetFrame()  # get the frame
        marker_names = client.GetMarkerNames(subjectName)
        marker_names = [x[0] for x in marker_names]

        if 'RTOE' not in marker_names or 'RHEE' not in marker_names or 'RANK' not in marker_names:
            print("Missing markers or marker name, please check the VICON software")
            sys.exit()
        
        # Base angles
        deg_15_in = client.GetMarkerGlobalTranslation(subjectName, 'deg_15_in')[0]
        deg_10_in = client.GetMarkerGlobalTranslation(subjectName, 'deg_10_in')[0]
        deg_5_in = client.GetMarkerGlobalTranslation(subjectName, 'deg_5_in')[0]
        deg_0 = client.GetMarkerGlobalTranslation(subjectName, 'deg_0')[0]
        deg_5_out = client.GetMarkerGlobalTranslation(subjectName, 'deg_5_out')[0]
        deg_10_out = client.GetMarkerGlobalTranslation(subjectName, 'deg_10_out')[0]
        deg_15_out= client.GetMarkerGlobalTranslation(subjectName, 'deg_15_out')[0]

        # Foot markers
        RTOE_manual = client.GetMarkerGlobalTranslation(subjectName, 'RTOE')[0]
        RHEE_manual = client.GetMarkerGlobalTranslation(subjectName, 'RHEE')[0]
        RANK_manual = client.GetMarkerGlobalTranslation(subjectName, 'RANK')[0]

        # add error exception for occluded markers
        if RTOE_manual == [0, 0] or RHEE_manual == [0, 0]:
            # Flag  data 
            print("Too many occlusions for RHEE/RTOE, check the markers")
            FPA_manual = np.nan
        else:
            # Calculate FPA
            footVec_manual = (RTOE_manual[0] - RHEE_manual[0], RTOE_manual[1] - RHEE_manual[1])
            FPA_manual = -math.degrees(math.atan(footVec_manual[1] / footVec_manual[0])) 

        ################# Allowing the participant to move foot to the desired angle #################
             
        print("        Press space after allowing the participant to move their own foot to " + str(deg_test[deg_i]) + " deg: ")
        keyboard.wait('space')  
    
        client.GetFrame() 
        
        # Foot markers
        RTOE_prop = client.GetMarkerGlobalTranslation(subjectName, 'RTOE')[0]
        RHEE_prop = client.GetMarkerGlobalTranslation(subjectName, 'RHEE')[0]

        # add error exception for occluded markers
        if RTOE_prop == [0, 0] or RHEE_prop == [0, 0]:
            # Flag data 
            print("Too many occlusions for RHEE/RTOE, check the markers")
            FPA_prop = np.nan
        else:
            # Calculate FPA
            footVec_prop = (RTOE_prop[0] - RHEE_prop[0], RTOE_prop[1] - RHEE_prop[1])
            FPA_prop = -math.degrees(math.atan(footVec_prop[1] / footVec_prop[0]))

        if deg_test[deg_i] > 0:
            err_prop.append(abs(FPA_prop - FPA_manual))
            err_prop_out.append(err_prop[deg_i])
        elif deg_test[deg_i] < 0:
            err_prop.append(abs(FPA_prop - FPA_manual)) 
            err_prop_in.append(err_prop[deg_i])
        
        print("                The error for this trial was: " + str(err_prop[deg_i]))
        FPA_store.append((time.time_ns(), deg_test[deg_i], FPA_manual, FPA_prop, err_prop[deg_i], RHEE_manual[0], RHEE_manual[1], RHEE_prop[0], RHEE_prop[1], RTOE_manual[0], RTOE_manual[1], RTOE_prop[0], RTOE_prop[1])) 
        base_angle_store.append((time.time_ns(), deg_test[deg_i], deg_15_in[0], deg_15_in[1], deg_10_in[0], deg_10_in[1], deg_5_in[0], deg_5_in[1], deg_0[0], deg_0[1], deg_5_out[0], deg_5_out[1], deg_10_out[0], deg_10_out[1], deg_15_out[0], deg_15_out[1]))        

    # save calculated FPA
    df_main = pd.DataFrame(FPA_store, columns = ['time (ns)', 'deg test', 'FPA manual (deg)', 'FPA prop (deg)', 'absolute error (deg)', 'RHEE manual x-', 'RHEE manual y-', 'RHEE proprio x-', 'RHEE proprio y-', 'RTOE manual x-', 'RTOE manual y-', 'RTOE proprio x-', 'RTOE proprio y-'])
    df_base = pd.DataFrame(base_angle_store, columns = ['time (ns)', 'deg test', 'deg_15_in_x', 'deg_15_in_y', 'deg_10_in_x', 'deg_10_in_y', 'deg_5_in_x', 'deg_5_in_y', 'deg_0_x', 'deg_0_y', 'deg_5_out_x', 'deg_5_out_y', 'deg_10_out_x', 'deg_10_out_y', 'deg_15_out_x', 'deg_15_out_y'])
    df_main.to_csv(csv_file_main)
    df_base.to_csv(csv_file_baseAngle)
    
    # print avg error
    print("**---------------------------------------------------------------------------------------------------**")
    print("        Average error: " + str(round(np.nanmean(err_prop),2)) + "  +/-  " + str(round(np.nanstd(err_prop),2)))
    print("        Average toe-in error: " + str(round(np.nanmean(err_prop_in),2)) + "  +/-  " + str(round(np.nanstd(err_prop_in),2)))
    print("        Average toe-out error: " + str(round(np.nanmean(err_prop_out),2)) + "  +/-  " + str(round(np.nanstd(err_prop_out),2)))
    print("**---------------------------------------------------------------------------------------------------**")
        
except ViconDataStream.DataStreamException as e:
    print( 'Handled data stream error: ', e )
except KeyboardInterrupt:
    print( 'Keyboard interrupt detected, trial ended early and data was not saved' )