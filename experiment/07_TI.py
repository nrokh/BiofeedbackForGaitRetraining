from vicon_dssdk import ViconDataStream
import argparse
import sys
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keyboard
import asyncio
import struct
from bleak import BleakScanner, BleakClient
import tkinter as tk
from tkinter import filedialog
import os

############# GAIT GUIDE SETUP #####################

# gaitguide BLE settings:
BLE_DURATION_STIM_SERVICE_UUID = '1111'
BLE_AMPLITUDE_CHARACTERISTIC_UUID = '1112' 
BLE_DURATION_RIGHT_CHARACTERISTIC_UUID = '1113' 
BLE_DURATION_LEFT_CHARACTERISTIC_UUID = '1114'  
BLE_BATTERY_SERVICE_UUID = '180F'
BLE_BATTERY_LEVEL_CHARACTERISTIC_UUID = '2A19'
timeout = 5

async def connect_to_device():
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == 'GaitGuide':
            print('Device found - MAC [', d.address, ']')
            client = BleakClient(d.address)
            await client.connect(timeout=timeout)
            print('Connected [', d.address, ']')
            return client

def get_characteristic(service, characteristic_uuid):
    characteristic = service.get_characteristic(characteristic_uuid)
    return characteristic

async def set_amp(client, characteristic, value):
    await client.write_gatt_char(characteristic,  bytearray([value]))

async def write_characteristic(client, characteristic, value):
    await client.write_gatt_char(characteristic, bytearray(value))

async def read_characteristic(client, characteristic):
    value = await client.read_gatt_char(characteristic)
    return value

############# FILE SAVING #####################

def generate_csv_filename(directory, subject_name, parameter):
    
    csv_file = os.path.join(directory, subject_name[0] + '_' + parameter + '_1.csv')
    counter = 1

    while os.path.exists(csv_file):
        counter += 1
        csv_file = os.path.join(directory, subject_name[0] + '_' + parameter + '_' + str(counter) + '.csv')
    print('      Data will be saved to file: ', csv_file)
    return csv_file
    
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

    # Get the desired directory to save the data
    root = tk.Tk()
    root.withdraw() 
    directory = filedialog.askdirectory()

    # Connect to Bluetooth
    print(' Connecting to GaitGuide...')
    GaitGuide = asyncio.run(connect_to_device()) 

    print('Getting GaitGuide service...')
    service = GaitGuide.services.get_service(BLE_DURATION_STIM_SERVICE_UUID)

    if service:
        print('Setting Amp, Right and Left GaitGuide characteristics...')
        Right = get_characteristic(service, BLE_DURATION_RIGHT_CHARACTERISTIC_UUID)
        Left = get_characteristic(service, BLE_DURATION_LEFT_CHARACTERISTIC_UUID)
        Ampl = get_characteristic(service, BLE_AMPLITUDE_CHARACTERISTIC_UUID)

    # GG battery level checks:    
    print('Getting GG battery service...')
    BAT_service = GaitGuide.services.get_service(BLE_BATTERY_SERVICE_UUID)

    if BAT_service:
        Bat = get_characteristic(BAT_service, BLE_BATTERY_LEVEL_CHARACTERISTIC_UUID)

    batteryLevel = asyncio.run(read_characteristic(GaitGuide, Bat))
    batteryLevel_int = int.from_bytes(batteryLevel, "little")  
    print(f'Bat_Level = [{batteryLevel_int}%]')

    print('Setting GaitGuide amplitude to max...')
    asyncio.run(set_amp(GaitGuide, Ampl, 127))

    # create a list to store FPA and marker values
    FPA_store = []
    CAL_store = []
    PSI_store = []
    DIFF_store = [0,0,0]
    DIFFDV_store = [0,0,0] 
    gaitEvent_store = []
    FPAstep_store = []
    meanFPAstep_store = []  

    # create flag to check for systemic occlusions
    occl_flag_foot = 0 
    occl_flag_hip = 0

    # variables for the catch trials
    step_count = 0 
    cadence = 80 # steps per minute
    trial_time = 5 # minutes
    catch_steps = 40 # number of steps for the catch trial
    catch_flag = 0

    ############## SCALED FEEDBACK SETUP ###############
    band = 2 #degrees to either side

    feedbackType = float(input("Select feedback type: (0) = no feedback; (1) = trinary; (2) = scaled: "))
    
    if feedbackType == 0.0:
        print("Starting no feedback mode...")
    elif feedbackType == 1.0:
        print("Starting trinary feedback mode...")
    elif feedbackType == 2.0:
        print("Starting scaled feedback mode...")

    ################# ENTER BASELINE FPA ###############
    baselineFPA = float(input("Enter subject's baseline FPA and hit enter: "))
    targetFPA = baselineFPA - 10.0
    print("Target toe-in angle is: ", targetFPA)

    ################# STEP DETECTION ###################
    print("Press space when ready to start step detection: ")
    keyboard.wait('space')        

    local_max_detected = False

    while step_count < trial_time*cadence: 
        subjectName = subjectNames[0] # select the main subject
        client.GetFrame() # get the frame
        marker_names = client.GetMarkerNames(subjectName)
        marker_names = [x[0] for x in marker_names]

        ################# CALCULATE FPA ####################
        
        #check if all the main markers are streaming properly 
        if 'RTOE' not in marker_names or 'RHEE' not in marker_names or 'RPSI' not in marker_names:
            print("Missing markers or marker name, please check the VICON software")
            sys.exit()

        RTOE_translation = client.GetMarkerGlobalTranslation(subjectName, 'RTOE')[0]
        RHEE_translation = client.GetMarkerGlobalTranslation(subjectName, 'RHEE')[0]
        CAL = RHEE_translation[0]
        PSI = client.GetMarkerGlobalTranslation( subjectName, 'RPSI')[0][0]

        # add error exception for occluded markers
        if RTOE_translation == [0, 0] or RHEE_translation == [0, 0]:
            # Flag this data and check if it's consecutively too frequent
            occl_flag_foot += 1
            if occl_flag_foot > 25:
                print("Too many occlusions for RHEE/RTOE, check the markers")
            # Save FPA as a NaN value so we can discard later
            FPA = np.nan
        else:
            # Calculate FPA
            occl_flag_foot = 0
            footVec = (RTOE_translation[0] - RHEE_translation[0], RTOE_translation[1] - RHEE_translation[1])
            FPA = -math.degrees(math.atan(footVec[1] / footVec[0])) 
            CAL_store.append(CAL)

        # get AP CAL and PSI markers  
        if PSI == 0:
            occl_flag_hip += 1
            if occl_flag_hip > 25:
                print("Too many occlusions for PSI, check marker") 
        
        # take derivative of difference between heel and hip:
        DIFF = CAL - PSI
        DIFF_store.append(DIFF)
        DIFFDV = DIFF_store[-1] - DIFF_store[-2] 
        DIFFDV_store.append(DIFFDV)

        # search for local max 
        if DIFFDV_store[-1]>=0 and DIFFDV_store[-2]<=0 and DIFFDV_store[-3]<=0 and DIFFDV_store[-4]<=0:

            print("local max")
            FPAstep_store = []
            local_max_detected = True
            gaitEvent_store.append((time.time_ns(), 1.0))

        FPAstep_store.append(FPA)

        # search for min:
        if local_max_detected and DIFFDV_store[-1]<=0 and DIFFDV_store[-2]>=0 and DIFFDV_store[-3]>=0 and DIFFDV_store[-4]>=0:
            print("local min")
            meanFPAstep = np.nanmean(FPAstep_store)
            meanFPAstep_store.append((time.time_ns(), meanFPAstep)) 
            print("mean FPA for step = " + str(meanFPAstep))
            gaitEvent_store.append((time.time_ns(), 2.0))
            step_count += 2
            # print(step_count)
            local_max_detected = False

            ################# CUE GAITGUIDE ###############
    
            if step_count >= ((cadence*trial_time/2)-(catch_steps/2)) and step_count <= ((cadence*trial_time/2)+(catch_steps/2)): 
                if feedbackType == 1.0 or feedbackType == 2.0:
                    print("Catch trial")
                    catch_flag = 1
                else:
                    catch_flag = 0
            else:
                catch_flag = 0

            if feedbackType == 0.0 and catch_flag == 0: #no feedback mode
                gaitEvent_store.append((time.time_ns(), 0.0, 'none', 0)) #for consistancy in logging the data

            elif feedbackType == 1.0 and catch_flag == 0: #trinary mode
                if meanFPAstep < targetFPA - band: # too far in
                    duration = 300
                    duration_packed = struct.pack('<H', int(duration))
                    asyncio.run(write_characteristic(GaitGuide, Right, duration_packed))
                    gaitEvent_store.append((time.time_ns(), 1.0, 'right', duration_packed))

                elif meanFPAstep > targetFPA + band: # too far out
                    duration = 300
                    duration_packed = struct.pack('<H', int(duration))
                    asyncio.run(write_characteristic(GaitGuide, Left, duration_packed))
                    gaitEvent_store.append((time.time_ns(), 1.0, 'left', duration_packed))

            elif feedbackType == 2.0 and catch_flag == 0: # scaled feedback mode
                if meanFPAstep < targetFPA - band: # too far in
                    duration = abs((targetFPA - meanFPAstep))*108 - 156
                    if duration > 600:
                        duration = 600
                    duration_packed = struct.pack('<H', int(duration))
                    asyncio.run(write_characteristic(GaitGuide, Right, duration_packed))
                    gaitEvent_store.append((time.time_ns(), 2.0, 'right', duration_packed))

                elif meanFPAstep > targetFPA + band: # too far out
                    duration = abs((targetFPA - meanFPAstep))*108 - 156
                    if duration > 600:
                        duration = 600
                    duration_packed = struct.pack('<H', int(duration))
                    asyncio.run(write_characteristic(GaitGuide, Left, duration_packed))
                    gaitEvent_store.append((time.time_ns(), 2.0, 'left', duration_packed))

        # save FPA value to the list
        FPA_store.append((time.time_ns(), FPA))

    GaitGuide.disconnect()

# save calculated FPA
    df_FPA = pd.DataFrame(FPA_store)
    csv_file_FPA = generate_csv_filename(directory, subjectNames, 'FPA')
    df_FPA.to_csv(csv_file_FPA)

    # save the mean FPA for each step w/ timestamps
    df_mFPA = pd.DataFrame(meanFPAstep_store)
    csv_file_mFPA = generate_csv_filename(directory, subjectNames, 'meanFPA')
    df_mFPA.to_csv(csv_file_mFPA)

    # save gait events
    df_GE = pd.DataFrame(gaitEvent_store)
    csv_file_GE = generate_csv_filename(directory, subjectNames, 'gaitEvents')
    df_GE.to_csv(csv_file_GE)

    # Save DIFF
    df = pd.DataFrame(DIFF_store)
    csv_file_df = generate_csv_filename(directory, subjectNames, 'DIFF')
    df.to_csv(csv_file_df) 

    # Save DIFFDV
    dfdv = pd.DataFrame(DIFFDV_store)
    csv_file_dfdv = generate_csv_filename(directory, subjectNames, 'DIFFDV')
    dfdv.to_csv(csv_file_dfdv)
    
    # Plot the FPA 
    plt.plot(df_FPA.iloc[:,0], df_FPA.iloc[:,1])
    plt.xlabel('Time [ns]')
    plt.ylabel('FPA [deg]')
    plt.scatter(df_mFPA.iloc[:,0], df_mFPA.iloc[:,1], color='red', marker='o')
    plt.title('FPA')
    plt.show()

except KeyboardInterrupt: # CTRL-C to exit    
    GaitGuide.disconnect()
    print('GaitGuide Disconnected [', GaitGuide.address, ']')
    
    # save calculated FPA
    df_FPA = pd.DataFrame(FPA_store)
    csv_file_FPA = generate_csv_filename(directory, subjectNames, 'FPA_Interrupted')
    df_FPA.to_csv(csv_file_FPA)

    # save the mean FPA for each step w/ timestamps
    df_mFPA = pd.DataFrame(meanFPAstep_store)
    csv_file_mFPA = generate_csv_filename(directory, subjectNames, 'meanFPA_Interrupted')
    df_mFPA.to_csv(csv_file_mFPA)

    # save gait events
    df_GE = pd.DataFrame(gaitEvent_store)
    csv_file_GE = generate_csv_filename(directory, subjectNames, 'gaitEvents_Interrupted')
    df_GE.to_csv(csv_file_GE)

    # save DIFF
    df = pd.DataFrame(DIFF_store)
    csv_file_df = generate_csv_filename(directory, subjectNames, 'DIFF_Interrupted')
    df.to_csv(csv_file_df) 

    # save DIFFDV
    dfdv = pd.DataFrame(DIFFDV_store)
    csv_file_dfdv = generate_csv_filename(directory, subjectNames, 'DIFFDV_Interrupted')
    dfdv.to_csv(csv_file_dfdv)
    
    # Plot the FPA 
    plt.plot(df_FPA.iloc[:,0], df_FPA.iloc[:,1])
    plt.xlabel('Time [ns]')
    plt.ylabel('FPA [deg]')
    plt.scatter(df_mFPA.iloc[:,0], df_mFPA.iloc[:,1], color='red', marker='o')
    plt.title('FPA')
    plt.show()

    print('Data saved!')

except ViconDataStream.DataStreamException as e:
    print( 'Handled data stream error: ', e )


