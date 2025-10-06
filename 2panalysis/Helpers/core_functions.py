#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 00:38:51 2018

@author: burakgur/catay aydin / Functions for 2 photon image and data 
    processing. Most functions are created by Catay Aydin.
"""

from __future__ import division
from enum import unique
try:
    from tkFileDialog import askdirectory, askopenfilename # For Python 2.X
except:
    from tkinter.filedialog import askdirectory, askopenfilename # For Python 3.X

from itertools import islice
from scipy.stats import pearsonr
from collections import Counter
import re
import os
import numpy
import glob
import multiprocessing
import pandas as pd
import numpy as np
from skimage import io
import warnings

def getDataDirectory(initialDirectory='~'):
    """Gets the raw data directory and analysis output directory paths.

    Parameters
    ==========

    initialDirectory : str, optional
        Default: HOME directory, i.e. '~'.

        Path into which the directory selection GUI opens.

    Returns
    =======
    rawDataDir : str
        Raw data path.

    rawDataFile : str
        Raw tif files' path.

    xmlFile : str
        XML file path.

    outDir : str
        Output diectory path.

    baseName : str
        Name of the time series folder.

    stimOutFile : str
        Stimulus output file path.
    """
    root = Tkinter.Tk()
    root.withdraw()

    rawDataDir = askdirectory(parent=root,
                              initialdir=initialDirectory,
                              title='Raw Data Directory')
    outDir = askdirectory(parent=root,
                          initialdir=initialDirectory,
                          title='Output Directory')

    rawDataFile = os.path.join(rawDataDir, '*.tif')
    stimOutPath = os.path.join(rawDataDir, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    baseName = os.path.basename(rawDataDir)
    # there might be other xml file in the directory
    # e.g. when you use markpoints
    # so use glob style regex to get only the main xml
    xmlPath = os.path.join(rawDataDir, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]

    return rawDataDir, rawDataFile, xmlFile, outDir, baseName, stimOutFile

def readStimOut(stimOutFile,Tseries_len,skipHeader,cplusplus=False):
    """Read and get the stimulus output data.

    Parameters
    ==========
    stimOutFile : str
        Stimulus output file path.

    skipHeader : int, optional
        Default: 1

        Number of lines to be skipped from the beginning of the stimulus
        output file.

    Returns
    =======
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file.

    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.
    """
    # skip the first line since it is a file path
    
    if cplusplus==True:
        skipHeader=1
        Stim_content=open(stimOutFile, "r")
        line=Stim_content.readlines()
        if 'nothing' in line[0] or 'Nothing' in line[0]:
            skipHeader=2
        rawStimData = numpy.genfromtxt(stimOutFile, dtype='float',
                                   skip_header=skipHeader)#,delimiter=',') #Juan comment: to deal with c++ outputfiles # Seb: delimiter = ',' for _stimulus_ouput from 2pstim
    else:
        Stim_content=open(stimOutFile, "r")
        keys=Stim_content.readlines()
        keys=keys[2].split('\n')[0].split(',')
        rawStimData = numpy.genfromtxt(stimOutFile, dtype='float',
                                   skip_header=skipHeader,delimiter=',')
    # also get the file path
    # do not mix it with numpy
    # only load and read the first line
    
    #erase any extra entries (entries that go beyond the number of frames), update made by Juan
    framefilter=rawStimData[:,-1]<=Tseries_len
    framefilter=np.sum(framefilter)
    rawStimData=rawStimData[:np.sum(framefilter),:]
    ##
    
    stimType = "stimType"
    with open(stimOutFile, 'r') as infile:
        if cplusplus==True:
            Stim_content=open(stimOutFile, "r")
            line=Stim_content.readlines()
            if 'nothing' in line[0] or 'Nothing' in line[0]:
                lines_gen = islice(infile, 2)
            else:              
                lines_gen = islice(infile, 1)

        else:
            lines_gen = islice(infile, 2) # Seb: the stimType is printed in line'2' of the _stimulus_ouput from 2pstim
        for line in lines_gen:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            line = re.sub(' ', '', line)
            stimType = line
            stimType=[line.split('\\')[-1]][0] #Juan this line deals with c++ stims
            
            # break #Seb: commented this out

    return stimType, rawStimData, keys

def readStimInput(stimType='', stimInputDir='', gui=True,
                  initialDirectory='~'): #NOT USED by BURAK
    """
    Parameters
    ==========
    stimType : str, optional
        Default: ''
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str, optional
        Default: ''

        Path of the directory to look for the stimulus input.

    gui : bool, optional
        Default: False

        Whether to use GUI for directory selection or not.

    initialDirectory : str, optional
        Default: HOME directory, i.e. '~'.

        Path into which the directory selection GUI opens.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).

    Notes
    =====
    This function does not return the values below anymore:

    - epochDur : list
        Duration of the epochs as in the stimulus input file.

    - isRandom : int

    - stimTransAmp : list

    """

    if stimInputDir == '' and not gui:
        print("Give a value to stimInputDir XOR gui")
        return None, None, None, None, None

    elif stimInputDir != '' and gui:
        print("Cannot give a non-default value to both gui and stimInputDir"
              "at the time ")
        return None, None, None, None, None

    else:
        if stimInputDir == '' and gui:
            initialDirectory = makePath(initialDirectory)
            root = Tkinter.Tk()
            root.withdraw()

            stimInputFile = askopenfilename(parent=root,
                                            initialdir=initialDirectory,
                                            filetypes=[('Text', '*.txt')],
                                            title='Open stimInputFile')

        elif stimInputDir != '' and not gui:
            stimInputDir = makePath(stimInputDir)
            stimType = stimType.split('\\')[-1]
            stimInputFile = glob.glob(os.path.join(stimInputDir, stimType))[0]
        # @TODO: Recognize if there is a slash or not after stimInputDir

        flHandle = open(stimInputFile, 'r')
        stimInputData = {}
        for line in flHandle:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            line = line.split('\t')
            stimInputData[line[0]] = line[1:]

#        epochDur = stimInputData['Stimulus.duration']
#        isRandom = int(stimInputData['Stimulus.randomize'][0])
#        stimTransAmp = stimInputData['Stimulus.stimtrans.amp']
#
#        epochDur = [float(sec) for sec in epochDur]
#        stimTransAmp = [float(deg) for deg in stimTransAmp]

        return stimInputFile, stimInputData


def getEpochCount(rawStimData, epochColumn=3):
    """Get the total epoch number.

    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    Returns
    =======
    epochCount : int
        Total number of epochs.
    """
    # get the max epoch count from the rawStimData
    # 4th column is the epoch number
    # add plus 1 since the min epoch no is zero
    
    # BG edit: Changed the previous epoch extraction, which uses the maximum 
    # number + 1 as the epoch number, to a one finding the unique values and 
    # taking the length of it
    epochCount = np.shape(np.unique(rawStimData[:, epochColumn]))[0]
    print("Number of epochs = " + str(epochCount))

    return epochCount


def readStimInformation(stimType, original_stimDir):
    """
    Parameters
    ==========
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str
        Path of the directory to look for the stimulus input.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).
    
    isRandom : int
        A that changes how epochs are randomized 

    Notes
    =====
    This function does not return the values below anymore:

    - epochDur : list
        Duration of the epochs as in the stimulus input file.

    - stimTransAmp : list

    """
    stimType = stimType.split('/')[-1] # Sen: \\ to / 
    if 'Search' in stimType:
        stimType=stimType.split('Search_')[-1]
    stimInputFile = glob.glob(os.path.join(original_stimDir, stimType))[0] #Juan: names changed
    

    # Seb: commnted out this and replaced for waht is below
    # flHandle = open(stimInputFile, 'r')
    # stimInputData = {}
    # for line in flHandle:
    #     line = re.sub('\n', '', line)
    #     line = re.sub('\r', '', line)
    #     line = line.split('\t')
    #     stimInputData[line[0]] = line[1:]

    
    ###################################### Seb version ###################################
    # To avoid that extra invisible 'tabs' end up in the stimInputFile dictionary
    stimInputData = {}
    with open(stimInputFile) as file:
        for line in file:
            curr_list = line.split()

            if not curr_list:
                 continue
                    
            key = curr_list.pop(0)
                
            if len(curr_list) == 1 and not "Stimulus." in key:
                try:
                    stimInputData[key] = int(curr_list[0])
                except ValueError:
                    stimInputData[key] = curr_list[0]
                continue 
                    
            if key.startswith("Stimulus."):
                key = key[9:]
                    
                if key.startswith("stimtype"):
                    stimInputData[key] = list(map(str, curr_list))
                else: 
                    stimInputData[key] = list(map(float, curr_list))

#########################################################################################

#        epochDur = stimInputData['Stimulus.duration']
    
#        stimTransAmp = stimInputData['Stimulus.stimtrans.amp']
#
#        epochDur = [float(sec) for sec in epochDur]
#        stimTransAmp = [float(deg) for deg in stimTransAmp]

    return stimInputFile, stimInputData

def divideEpochs(rawStimData, epochCount, isRandom, framePeriod,
                 trialDiff=0.20, overlappingFrames=0, firstEpochIdx=0,
                 epochColumn=3, imgFrameColumn=7, incNextEpoch=True,
                 checkLastTrialLen=True):
    """
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    isRandom : int

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    overlappingFrames : int

    firstEpochIdx : int
        Default: 0

        Index of the first epoch.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    incNextEpoch :
    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [Z, D]] where first term is the trial beginning
        (first of first) and end (second of first), and second term is the
        baseline start (first of second) and end (second of second) for that
        trial.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).

    isRandom :
    """
    trialDiff = float(trialDiff)
    firstEpochIdx = int(firstEpochIdx)
    overlappingFrames = int(overlappingFrames)
    trialCoor = {}
    fullEpochSeq = []

    if isRandom == 0:
        fullEpochSeq = range(epochCount)
        # if the key is zero, that means isRandom is 0
        # this is for compatibitibility and
        # to make unified workflow with isRandom == 1
        trialCoor[0] = []

    elif isRandom == 1:
        # in this case fullEpochSeq is just a list of dummy values
        # important thing is its length
        # it's set to 3 since trials will be sth like: 0, X
        # if incNextEpoch is True, then it will be like : 0,X,0
        fullEpochSeq = list(range(2))
        for epoch in range(1, epochCount):
            # add epoch numbers to the dictionary
            # do not add the first epoch there
            # since it is not the exp epoch
            # instead it is used for baseline and inc coordinates
            trialCoor[epoch] = []

    if incNextEpoch:
        # add the first epoch
        fullEpochSeq.append(firstEpochIdx)
    elif not incNextEpoch:
        pass

    # min and max img frame numbers for each and every trial
    # first terms in frameBaselineCoor are the trial beginning and end
    # second terms are the baseline start and end for that trial
    currentEpochSeq = []
    frameBaselineCoor = [[0, 0], [0, 0]]
    nextMin = 0
    baselineMax = 0

    for line in rawStimData:
        if (len(currentEpochSeq) == 0 and
                len(currentEpochSeq) < len(fullEpochSeq)):
            # it means it is the very beginning of a trial block.
            # in the very first trial,
            # min frame coordinate cannot be set by nextMin.
            # this condition satisfies this purpose.
            currentEpochSeq.append(int(line[epochColumn]))
            if frameBaselineCoor[0][0] == 0:
                frameBaselineCoor[0][0] = int(line[imgFrameColumn])
                frameBaselineCoor[1][0] = int(line[imgFrameColumn])

        elif (len(currentEpochSeq) != 0 and
              len(currentEpochSeq) < len(fullEpochSeq)):
            # only update the current epoch list
            # already got the min coordinate of the trial
            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for the very first trial
                    frameBaselineCoor[1][1] = (int(line[imgFrameColumn])
                                               - overlappingFrames)

        elif len(currentEpochSeq) == len(fullEpochSeq):
            if nextMin == 0:
                nextMin = int(line[imgFrameColumn]) + overlappingFrames

            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for all the trials except the very first trial
                    baselineMax = int(line[imgFrameColumn]) - overlappingFrames

        else:
            frameBaselineCoor[0][1] = (int(line[imgFrameColumn])
                                       - overlappingFrames)

            if frameBaselineCoor[0][1] > 0:
                if isRandom == 0:
                    # if the key is zero, that means isRandom is 0
                    # this is for compatibitibility and
                    # to make unified workflow isRandom == 1
                    trialCoor[0].append(frameBaselineCoor)
                elif isRandom == 1:
                    # get the epoch number
                    # epoch no should be the 2nd term in currentEpochSeq
                    expEpoch = currentEpochSeq[1]
                    trialCoor[expEpoch].append(frameBaselineCoor)
            # this is just a safety check
            # towards the end of the file, the number of epochs might
            # not be enough to form a trial block
            # so if the max img frame coordinate is still 0
            # it means this not-so-complete trial will be discarded
            # only complete trials are appended to trial coordinates
            # if it has a max frame coord, it is safe to say
            # it had nextMin in frameBaselineCoor
            # print(currentEpochSeq)
            currentEpochSeq = []
            currentEpochSeq.append(firstEpochIdx)
            # each time currentEpochSeq resets means that
            # one trial block is complete
            # adding firstEpochIdx is necessary
            # otherwise currentEpochSeq will shift by 1
            # after every trial cycle
            # now that the frame coordinates are stored
            # can reset it
            # and add min coordinate for the next trial
            # then add the max baseline coordinate for the next trial
            frameBaselineCoor = [[0, 0], [0, 0]]
            frameBaselineCoor[0][0] = nextMin
            frameBaselineCoor[1][0] = nextMin
            frameBaselineCoor[1][1] = baselineMax
            nextMin = 0
            baselineMax = 0

    # @TODO: no need to separate isRandoms, make a unified for loop
    if checkLastTrialLen:
        if isRandom == 0:
            lenFirstTrial = trialCoor[0][0][0][1] - trialCoor[0][0][0][0]
            lenLastTrial = trialCoor[0][-1][0][1] - trialCoor[0][-1][0][0]
            if ((lenFirstTrial - lenLastTrial) * framePeriod) >= trialDiff:
                trialCoor[0].pop(-1)
                print("Last trial is discarded since the length was too short")

        elif isRandom == 1:
            for epoch in trialCoor:
                delSwitch = False
                lenFirstTrial = (trialCoor[epoch][0][0][1]
                                 - trialCoor[epoch][0][0][0])
                lenLastTrial = (trialCoor[epoch][-1][0][1]
                                - trialCoor[epoch][-1][0][0])

                if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                    delSwitch = True

                if delSwitch:
                    print("Last trial of epoch " + str(epoch)
                          + " is discarded since the length was too short")
                    trialCoor[epoch].pop(-1)

    trialCount = []
    if isRandom == 0:
        # there is only a single key in the trialCoor dict in this case
        trialCount.append(len(trialCoor[0]))
    elif isRandom == 1:
        epochTrial = []
        for epoch in trialCoor:
            epochTrial.append(len(trialCoor[epoch]))
        # in this case first element in trialCount is min no of trials
        # second element is the max no of trials
        trialCount.append(min(epochTrial))
        trialCount.append(max(epochTrial))

    return trialCoor, trialCount, isRandom

def divide_trials_1epoch(arr):
    # Input validation
    if not isinstance(arr, np.ndarray) or arr.shape[1] != 8:
        raise ValueError("Input must be a NumPy array with shape (n, 8)")
    
    # Initialize the output list
    output = []
    
    # Get unique values in the third column, preserving order
    unique_vals, indices = np.unique(arr[:, 2], return_index=True)
    indices = np.sort(indices)  # Ensure indices are sorted
    
    if len(unique_vals) == 1:
        return {0:[[int(arr[0][-1]),-1],[int(arr[0][-1]),-1]]}, 1
        # trialCount = 1

    # Iterate through unique values
    for i in range(len(indices)):
        uniq_val = unique_vals[i]
        start_index = indices[i]
        if (i + 1) < len(indices):
            end_index = indices[i + 1] - 1  # -1 to adjust for the next start
        else:
            end_index = -1  # Handle the last segment
            end_index = -1
        # Extract the start and end value from the eighth column
        start_val = int(arr[start_index, 7])
        end_val = int(arr[end_index, 7] if end_index != -1 else arr[-1, 7])
        
        output.append([start_val, end_val])
    return {0:output}, unique_vals

def divide_all_epochs(rawStimData, epochCount, framePeriod, trialDiff=0.20,
                      epochColumn=3, imgFrameColumn=7,checkLastTrialLen=True):
    """
    
    Finds all trial and epoch beginning and end frames
    
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [X, Y]] where X is the trial beginning and Y 
        is the trial end.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).
    """
    trialDiff = float(trialDiff)
    trialCoor = {}
    
    for epoch in range(0, epochCount):
        
        trialCoor[epoch] = []

    previous_epoch = []
    for line in rawStimData:
        
        current_epoch = int(line[epochColumn])
        
        if (not(previous_epoch == current_epoch )): # Beginning of a new epoch trial
            
            
            if (not(previous_epoch==[])): # If this is after stim start (which is normal case)
                epoch_trial_end_frame = previous_frame
                trialCoor[previous_epoch].append([[epoch_trial_start_frame, epoch_trial_end_frame], 
                                            [epoch_trial_start_frame, epoch_trial_end_frame]])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                previous_epoch = int(line[epochColumn])
                
            else:
                previous_epoch = int(line[epochColumn])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                
        previous_frame = int(line[imgFrameColumn])
        
    if checkLastTrialLen:
        for epoch in trialCoor:
            delSwitch = False
            lenFirstTrial = (trialCoor[epoch][0][0][1]
                             - trialCoor[epoch][0][0][0])
            lenLastTrial = (trialCoor[epoch][-1][0][1]
                            - trialCoor[epoch][-1][0][0])
    
            if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                delSwitch = True
    
            if delSwitch:
                print("Last trial of epoch " + str(epoch)
                      + " is discarded since the length was too short")
                trialCoor[epoch].pop(-1)
                
    trialCount = []
    epochTrial = []
    for epoch in trialCoor:
        epochTrial.append(len(trialCoor[epoch]))
    # in this case first element in trialCount is min no of trials
    # second element is the max no of trials
    trialCount.append(min(epochTrial))
    trialCount.append(max(epochTrial))
       

    return trialCoor, trialCount

def dff(trialCoor, header, bgIndex, bgSub, baselineEpochPresent, baseDur):
    """ Calculate df/f. Modified by burak to handle stimuli without a baseline
    epoch. This function calculates dF/F for each epoch and each trial within 
    the epoch seperately.

    Parameters
    ==========
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch.
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.

    header : list
        It is a list of ROI labels, in the order it appears in the extracted
        signal file.

    bgIndex : int
        Index of the background in the header.

    bgSub : ndarray
        Contains the background-subracted intensities for all ROIs, including
        the background ROI.
        
    baselineEpochPresent: bool
        If the baseline epoch present or not. If yes, this epoch will be used
        for dF/F calculations.
    
    baseDur : int
        Frame numbers before the actual epoch to take as baseline

    Returns
    =======
    dffTraceAllRoi : dict
        df/f trace of all the ROIs. Each key is an epoch number. Corresponding
        value is a list of lists. Every element in the outer list (i.e. the
        inner list elements) corresponds to ROIs, in the same order as in the
        header. Every 'ROI list' is a list of numpy arrays, where each array
        correponds to a trial.. Note that the background ROI has NaN values
        for every trial.

    baselineStdAllRoi : dict
        Standard deviation of the baseline trace. Each key is an epoch number.
        Corresponding value is a list of lists. Every element in the outer
        list (i.e. the inner list elements) corresponds to ROIs, in the same
        order as in the header. 'ROI lists' have floats which correpond to
        standard deviation of each trial.

    baselineMeanAllRoi : dict
        Mean of the baseline trace. Each key is an epoch number.
        Corresponding value is a list of lists. Every element in the outer
        list (i.e. the inner list elements) corresponds to ROIs, in the same
        order as in the header. 'ROI lists' have floats which correpond to
        mean of each trial.
    """

    dffTraceAllRoi = {}
    baselineStdAllRoi = {}
    baselineMeanAllRoi = {}
    # in the first iteration, each element in dffTraceAllRoi,
    # is an roi; in each roi, single elements are trials
    # IMPORTANT NOTE: Calculations are fast numpy operations,
    # but eventually calculated values values go into a list,
    # which is then converted to a numpy object-not an array.
    # the reason is that not all trial blocks have the same length.
    # right now it benefits from fast numpy calc, but in the future
    # you might want to do it more elegantly.
    # bg roi is filled by NaNs, and there is only one NaN per trial
    for epochNo in trialCoor:
        epochNo = int(epochNo)  # just to be safe
        dffTraceAllRoi[epochNo] = []  # init an empty list with epochNo key
        baselineStdAllRoi[epochNo] = []
        baselineMeanAllRoi[epochNo] = []
        for roiIdx in range(len(header)):
            for trial in trialCoor[epochNo]:
                # take this many rows of a particular column
                # -1 is bcs img frame indices start from 1 in trialCoor
                # run away from the division by zero problem for background
                normTrace = numpy.float(0)
                if roiIdx == bgIndex:
                    normTrace = numpy.nan
                    normBaseline = numpy.nan
                    normBaselineStd = numpy.nan
                else:
                    if baselineEpochPresent: # Take the baseline as F0
                        trace = bgSub[trial[0][0]-1:trial[0][1], roiIdx]
                        baseTrace = bgSub[trial[1][0]-1:trial[1][1], roiIdx]
                        baseline = numpy.average(baseTrace[-baseDur:])
                        normTrace = (trace - baseline) / baseline
                        normBaseTrace = (baseTrace - baseline) / baseline
                        # calculate baseline stdev
                        # might need for thresholding in the future
                        normBaseline = numpy.average(normBaseTrace)
                        normBaselineStd = numpy.std(normBaseTrace)
                    else: # Taking the mean of all trial as F0
                        trace = bgSub[trial[0][0]-1:trial[0][1], roiIdx]
                        baseline = numpy.average(trace)
                        normTrace = (trace - baseline) / baseline
                        normBaseline = numpy.nan # Returns NaN
                        normBaselineStd = numpy.nan # Returns NaN
                try:
                    dffTraceAllRoi[epochNo][roiIdx].append(normTrace)
                    baselineStdAllRoi[epochNo][roiIdx].append(normBaselineStd)
                    baselineMeanAllRoi[epochNo][roiIdx].append(normBaseline)
                except IndexError:
                    dffTraceAllRoi[epochNo].append([normTrace])
                    baselineStdAllRoi[epochNo].append([normBaselineStd])
                    baselineMeanAllRoi[epochNo].append([normBaseline])

    return dffTraceAllRoi, baselineStdAllRoi, baselineMeanAllRoi
def trialAverage(dffTraceAllRoi, bgIndex):
    """ Take the average of df/f traces across trials.

    Paremeters
    ==========
    dffTraceAllRoi : dict
        df/f trace of all the ROIs. Each key is an epoch number. Corresponding
        value is a list of lists. Every element in the outer list (i.e. the
        inner list elements) corresponds to ROIs, in the same order as in the
        header. Every 'ROI list' is a list of numpy arrays, where each array
        correponds to a trial.. Note that the background ROI has NaN values
        for every trial.

    bgIndex : int
        Index of the background in the header.

    Returns
    =======
    trialAvgAllRoi : dict
        Average of df/f traces across trials. Each key is an epoch number.
        Corresponding value is a list of numpy arrays. Each numpy array is an
        ROI and they are ordered in the same way as in the header. Background
        ROI has a single NaN value.
    """
    # each element in trialAvgAllRoi is an epoch
    # then each epoch has a single list
    # this list contains arrays of trial averages for every roi
    # if bg, instead of an array, it has NaN
    trialAvgAllRoi = {}
    for epoch in dffTraceAllRoi:
        trialAvgAllRoi[epoch] = []
        for roi in range(len(dffTraceAllRoi[epoch])):
            trialLengths = []
            # if Bg, append NaN to trialLengths
            # this way you dont distrupt bgIndex in the future
            if roi == bgIndex:
                trialLengths.append(numpy.nan)
            else:
                for trial in dffTraceAllRoi[epoch][roi]:
                    trialLengths.append(len(trial))

            if trialLengths[0] is not numpy.nan:
                # real ROI case, not bg
                minTrialLen = min(trialLengths)
                trialFit = 0
                for trial in dffTraceAllRoi[epoch][roi]:
                    trialFit += trial[:minTrialLen]
                # calculate the trial average for an roi
                trialAvg = trialFit/len(dffTraceAllRoi[epoch][roi])

                trialAvgAllRoi[epoch].append(trialAvg)

            elif trialLengths[0] is numpy.nan:
                # bgRoi case
                trialAvgAllRoi[epoch].append(numpy.nan)

    return trialAvgAllRoi

   
def interpolateTrialAvgROIs(trialAvgAllRoi, framePeriod, intRate):
    """ Interpolates the responses of ROIs that are trial averaged and sorted
    into epochs. 

    Parameters
    ==========
    trialAvgAllRoi : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which consists of numpy arrays and each array is the trial
        averaged trace of an ROI. ROIs are ordered in the same way as in the
        header, e.g. the first term in trialAvgAllRoi corrsponds to the first
        ROI label in the header.

    framePeriod : float
        Time it takes to image a single frame.

    intRate : int
        Interpolation rate in Hz.

    Returns
    =======
    interpolatedAllRoi : dict
        Same format as trialAvgAllRoi. Contains the interpolated arrays to the 
        desired frequency.

    """
    
    interpolatedAllRoi ={}
    for index, epoch in enumerate(trialAvgAllRoi): # For all epochs
        epochResponses = trialAvgAllRoi[epoch]
        interpolatedAllRoi[epoch] = []
        timeV = np.linspace(0,len(epochResponses[0]),len(epochResponses[0]))
        # Create an interpolated time vector in the desired interpolation rate
        timeVI = np.linspace(0,len(epochResponses[0]),
                             (len(epochResponses[0])*framePeriod*intRate+1))
        for iROI in range(len(epochResponses)): # For all ROIs
            currROIResponse = epochResponses[iROI]
            if currROIResponse is not np.nan:
                newCurrResponses = np.interp(timeVI, timeV, currROIResponse)
                interpolatedAllRoi[epoch].append(newCurrResponses)
            else:
                interpolatedAllRoi[epoch].append(np.nan)
    return interpolatedAllRoi

#def makePath(path):
#    """Make Windows and POSIX compatible absolute paths automatically.
#
#    Parameters
#    ==========
#    path : str
#
#    Path to be converted into Windows or POSIX style path.
#
#    Returns
#    =======
#   compatPath : str
#    """

#    compatPath = os.path.abspath(os.path.expanduser(path))

#    return compatPath

#def getVarNames(varFile='variablesToSave.txt'):
#    """ Read the variable names from a plain-text document. Then it is used to
#    save and load the variables by conserving the variable names, in other
#    functions. Whenever a new function is added, one should also add the stuff
#    it returns (assuming returned values are stored in the same variable names
#    as in the function definition) to the varFile.

#    Parameters
#    ==========
#    varFile : str, optional
#        Default: 'variablesToSave.txt'

#        Plain-text file from where variable names are read.

#    Returns
#    =======
#    varNames : list
#        List of variable names
#    """
#    # get the variable names
#    varFile = makePath(varFile)
#    workspaceVar = open(varFile, 'r')
#    varNames = []

#    for line in workspaceVar:
#        if line.startswith('#'):
#            pass
#        else:
#            line = re.sub('\n', '', line)
#            line = re.sub('', '', line)
#            line = re.sub(' ', '', line)
#            if line == '':
#                pass
#            else:
#                varNames.append(line)
#    workspaceVar.close()#
#
#    return varNames

def saveWorkspace(outDir, baseName, varDict, varFile='workspaceVar.txt',
                  extension='.pickle'):
    """ Save the variables that are present in the varFile. The file format is
    Pickle, which is a mainstream python format.

    Parameters
    ==========
    outDir : str
        Output diectory path.

    baseName : str
        Name of the time series folder.

    varDict : dict

    varFile : str, optional
        Default: 'workspaceVar.txt'

        Plain-text file from where variable names are read.

    extension : str, optional
        Default: '.pickle'

        Extension of the file to be saved.

    Returns
    =======
    savePath : str
        Path (inc. the filename) where the analysis output is saved.
    """

    # it is safer to get the variables from a txt
    # otherwise the actual session might have some variables
    # @TODO make workspaceFl path not-hardcoded
    print(varFile)
    varFile = makePath(varFile)
    varNames = getVarNames(varFile=varFile)
    workspaceDict = {}

    for variable in varNames:
        try:
            # only get the wanted var names from globals
            workspaceDict[variable] = varDict[variable]
        except KeyError:
            pass

    # open in binary mode and use highest cPickle protocol
    # negative protocol means highest protocol: faster
    # use cPickle instead of pickle: faster
    # C implementation of pickle
    savePath = os.path.join(outDir, baseName + extension)
    saveVar = open(savePath, "wb")
    cPickle.dump(workspaceDict, saveVar, protocol=-1)


    
    saveVar.close()

    return savePath

def loadWorkspace(workspaceFile='', gui=True,
                  initialDirectory='~', extension='*.pickle'):
    """Loads a binary pickle file, which is the workspace of the analysis.

    Parameters
    ==========
    workspaceFile : str, optional
        Default: ''

        If the *gui* argument is False, this path is used to load the
        workspace. Do *not* omit the extension when entering the file name.

    gui : bool,optional
        Default: True

        Whether to use GUI to load a file.

    initialDirectory : str, optional
        Default: HOME directory, i.e. '~'.

        Path into which the directory selection GUI opens.

    extension : str, optional
        Default: '.pickle'

        Extension of the file to be loaded. Applicable when *gui* is *True*.

    Returns
    =======
    workspace : dict
        Whole workspace which is saved after the analysis. Keys are variable
        names.
    """
    if gui and workspaceFile != '':
        print("gui cannot be True when a value is given to workspaceFile"
              "(other than an empty string)")
        return None

    elif not gui and workspaceFile == '':
        print('Gui is disabled and nothing given to workspaceFile'
              'Change either one of them')
        return None

    else:
        if gui and workspaceFile == '':
            initialDirectory = makePath(initialDirectory)
            root = Tkinter.Tk()
            root.withdraw()
            workspaceFile = askopenfilename(parent=root,
                                            initialdir=initialDirectory,
                                            filetypes=[(extension, extension)],
                                            title='Open saved workspace')
        elif not gui and workspaceFile != '':
            workspaceFile = makePath(workspaceFile)

        workspaceFile = open(workspaceFile, 'rb')
        workspace = cPickle.load(workspaceFile)
        workspaceFile.close()

        return workspace
    
def searchDatabase(metaDataBaseFile, conditions,var_from_database_to_select):
    """ Reads the meta database and extracts the imageIDs with the conditions of 
    interest. Also returns the flyIDs together with the imageIDs in a pandas 
    data frame.

    ==========
    metaDataBaseFile : str
        Full path of the meta database file including .txt
        
    conditions : dict
        A dictionary with keys (conditions) that are the columns of meta database 
        file and values that are the conditions of interest.
        
     var_from_database_to_select : list
    
        A list of strings that indicate the data variables from the meta database
        to be kept for further analysis.

    Returns
    =======
    data_to_select : pandas dataframe
        A pandas data frame which keeps the flyIDs and can be used to 
        select data since it also includes the data file names.
    """
    # Read meta data and find the indices of interest
    # The indices will be those that are in the dataframe so that the user can index
    # the data frame to extract current_exp_ID in order to load the dataset
    metaData_frame = pd.read_csv(metaDataBaseFile,sep='\t',header=0)
    positive_indices = {}
    for condition_indentifier, condition in conditions.items():
        if condition.startswith('#'):
            pass
        else:
            currIndices = metaData_frame\
            [metaData_frame[condition_indentifier]==condition].index.values
            positive_indices[condition_indentifier] = set(currIndices.flatten())
        
    common_indices = list(set.intersection(*positive_indices.values()))
    
    data_to_select = pd.DataFrame()
    for variable in var_from_database_to_select:
        data_to_select[variable] = metaData_frame[variable].iloc[common_indices]
        
    
    
    
    return data_to_select

def extract_metadata(experiment,home_path):

    """
    takes txt files from 2photon recordings from a specific experiment and puts
    all the info together in a single csv file    
    
    args:
        experiment(string): name of the folder with the recordings with experiments
    saves:
        csv file with metaadata information 
    """
#    try:
#        meta_table= pd.read_csv(("E:\\PhD\\experiments\\2p\\"+ experiment +"\\processed\\metadata.csv"),index='fly')  
#    except:
#        print('meta_file doesnt exist')
    meta_table=[]
#    finally:
#        pass
    
    #path='E:\\PhD\\experiments\\2p\\spont_act_descr\\raw'
    os.chdir(home_path + experiment+"\\raw")
    txt_list=glob.glob('*.txt')
    for txt in txt_list:
        #open txt
        met_file=open(txt,'r')
        content=met_file.read()
        met_file.close()
        content=content.split('<')
        for fly in content[:]:
            #read lines and extract
            print(txt)
            number_recordings=int(fly[fly.find('number_of_recordings:')+21])
            output=fly.split('\n')
            fly_content=list()
            #index={output[1]}
            for entry in output[:-1]:
                if len(entry)>0:
                    line0=entry.split(':')[0]
                    line1=entry.split(':')[1].split(',')  
                    if 'fly:' in entry:
                        line1=line1*number_recordings
                    col=pd.DataFrame({line0:line1})
                    fly_content.append(col)
            for i in range(len(fly_content)):
                if i==0:
                    fly_info=fly_content[i]
                else:
                    fly_info=fly_info.join(fly_content[i],how='outer')
            #fly_info=fly_info.set_index('fly')
            if len(meta_table)!=0:
                meta_table=meta_table.append(fly_info,ignore_index=True)
            else:
                meta_table=fly_info
    #meta_table=meta_table.set_index('fly')
                    #meta_table=meta_table.set_index('fly')
    meta_table.to_csv((home_path+experiment+"\\processed\\metadata.csv"))

def produce_Tseries_list(home_path, experiment,experimenter_code):
    """
    make list with Tseries paths
    
    arg:
        experiment(string): name of folder containing experiment
    returns:
        list: Tseries paths
    """
    # path_raw=home_path+ experiment +'\\' + 'raw'  
    path_raw = os.path.join(home_path, experiment, 'raw')
    pathfly='*'+ experimenter_code +'*' #_jv_ #_seb_
    Tser_path=os.path.join(path_raw,pathfly,'TSeries*', '')
    Tser_path=(glob.glob(Tser_path)) 
    return Tser_path
# def produce_Tseries_list(home_path,experiment,experimenter_code):
#     """
#     make list with Tseries paths
    
#     arg:
#         experiment(string): name of folder containing experiment
#     returns:
#         list: Tseries paths
#     """
#     path_raw=home_path+ experiment +'\\' + 'raw'
#     if os.path.isdir(path_raw):
#         #pathfly='*'+ '20210111_jv_fly3'+'*'+'\\'
        
#         pathfly='*'+ experimenter_code+'*'+'\\' 
#         Tser_path=os.path.join(path_raw,pathfly,'TSeries*\\')
#         Tser_path=(glob.glob(Tser_path)) 
        
#     return Tser_path 

def extract_fly_metadata(Tser):
    
    ''' takes in a Tseries path and searches for a csv file containing the experiment metadata of an experiment
        the information in the csv is loaded into memory, with keys corresponding to the columns of the csv file
        
        input: 
              Tser (str): path of the current time series being analyzed
        output:
              dictionary. keys are the column names, values are the corresponding metadata values
              for the specific recording put as input
        '''
    
    
    
    metadata_path=os.path.split(os.path.split(os.path.split(Tser)[0])[0])[0]
    metadata_path=os.path.split(metadata_path)[0]+'\\processed\\metadata_fixed.csv'
    fly=os.path.split(os.path.split(os.path.split(Tser)[0])[0])[1]
    Tser=os.path.split(os.path.split(Tser)[0])[1]
    # import metadata as pandas dataframe
    metadata=pd.read_csv(metadata_path,header=0,index_col=0,skipinitialspace=True)
    metadata=metadata.reset_index()
    fly_metadata=metadata.loc[metadata['date']==fly]
    fly_metadata=fly_metadata.loc[fly_metadata['recording']==Tser]
    fly_metadata=fly_metadata.to_dict(orient='list')
    return fly_metadata

def find_cycles(path):
  
    ''' finds how many cycles a recording has and their length '''
    
    tif_path=path+'\\*Cycle*.tif'
    tif_list=glob.glob(tif_path)
    cycles=[]
    for tif in tif_list:
        cycle_num=tif.split('\\')[-1]
        cycle_num=cycle_num.split('_')[1]
        cycles.append(cycle_num)
    
    cycles,lenghtCycles=np.unique(np.array(cycles),return_counts=True)
    number_cycles=len(cycles)
    return number_cycles,lenghtCycles
    
def filter_cycles(path,number_cycles):
  
    ''' finds which images (tifs) correspond to each cycle in a tseries recording 
        returns a list of lists containing file names '''
    
    cycle_list={}
    for cycle in range(1,number_cycles+1):
        #find tifs for each cycle
        cycle_list[cycle]=glob.glob(path +'\\'+'*_Cycle*'+str(cycle)+'_ch*')
    return cycle_list  

def find_stimfile_check_stim(exp,Tser,number_of_cycles,old=True): #repair this function. relying on time is no good
        """
        read tseries date and then read date of creation of xml file
        then select the corresponding stimulus
        
        returns:
            stimtype(list): string with the name of stimuli type inside stim_files
            stim_names(list): stimfiles names 
        """
        #####date sorting stopped for unreliability

        # path_raw=home_path+'\\'+ exp +'\\' + 'raw\\'
        # path=[]
        # path=dataDir+current_t_series+'.xml' 
        # date=current_exp_ID.split('_')[0]
        # xml_time=pathlib.Path(glob.glob(path)[0])
        # xml_time=pathlib.Path.stat(xml_time)[8]
        # ##search for correct stim_file
        # stim_path= '*stim*_%s_jv*\\'%(date)
        # stim_path=path_raw + stim_path
        # stim_path=stim_path + '_stimulus_output*'
        # stimfiles=glob.glob(stim_path)# !!!!!sorted
        # for number_stim,stim in enumerate(stimfiles):
        #     stim_p=pathlib.Path(stim)
        #     stimtime=pathlib.Path.stat(stim_p)[8]
        #     if '2021_3_8' in stim : # this corrects a time delay observed in sepcific dates
        #         stimtime=stimtime-3600
        #     #stim.split('.txt')[0].split('_')[-3:] #in case we need repairing this code
        #     difference=stimtime-xml_time
        #     correct_stimulus=abs(difference)<30
        #     if correct_stimulus:     
        #         Stim_content= open(stim, "r")
        #         line=Stim_content.readlines()
        #         stimtype=[line[1].split(r'/')[-1]]
        #         Stim_content.close()
        #         #stim_name=[os.path.split(stim)] #TODO check if the slice is [0] or [1]
        #         stim_name=[stim]
        #         if number_of_cycles==1: 
        #             return stimtype, stim_name
        #         elif number_of_cycles>1:
        #             slice_number=number_of_cycles-1
        #             extra_names=np.array(stimfiles[number_stim-slice_number:number_stim])
        #             stim_names=np.append(extra_names,stim_name)
        #             mult_stim_types=np.array([])
        #             for cycle in extra_names:
        #                 Stim_content_add= open(cycle, "r")
        #                 line_add=Stim_content_add.readlines()
        #                 stimtype_add=[line_add[1].split(r'/')[-1]]
        #                 Stim_content_add.close()
        #                 mult_stim_types=np.append(mult_stim_types,stimtype_add)
        #             mult_stim_types=np.append(mult_stim_types,stimtype)
        #             original_stims=[]
        #             #return mult_stim_types, stim_names

        stim_names=sorted(glob.glob(Tser +'*stim*'))
        stim_types=[]
        if len(stim_names)<number_of_cycles:
            raise Exception('at least 1 stim missing for %s'%(Tser))
        for stim in stim_names:
            Stim_content= open(stim, "r")
            line=Stim_content.readlines()
            if ('\\' in line[0] or 'nothing' in line[0] or 'Nothing' in line[0]):
                cplusplus=True
                stim_type=[line[0].split('\\')[-1]][0]
                stim_type=[stim_type.split(r'/')[-1]][0]
                if 'Search' in stim_type:
                    stim_type.split('Search_')[-1]
            else:
                stim_type=[line[1].split('\\')[-1]][0] #this line deals with c++ stims
                stim_type=[stim_type.split(r'/')[-1]][0]
                cplusplus=False

            stim_types.append(stim_type)
        return stim_types, stim_names,cplusplus

def load_Tseries(dataDir,number_of_cycles,lenghtCycles,file_string='_motCorr.tif'):
    
    ''' reads a multipage tif file containing a Tseries recording, divides it in cycles and stores the resulting tseries arrays in a list
        Tseries[cycle] with length = # of cycles 
        
        returns mean image and list of Tseries for each cycle '''
    
    tif_location=dataDir + file_string # if error here, check MC or non MC
    if len(glob.glob(tif_location))>1:
        raise Exception('more than one stack available for loading')
    time_series = io.imread(glob.glob(tif_location)[-1])
    mean_image = time_series.mean(0)
    #time_series = np.asarray(time_series) #make sure that the shape here is correct. it should be time in axis 0
    Tserlist=[]
    lenght_prev_cycles=0
    #TODO put time_series in sliceable format
    for cycle in range(number_of_cycles):
        currentNumberFrames=lenghtCycles[cycle]
        series_slice=time_series[lenght_prev_cycles:lenght_prev_cycles+currentNumberFrames,:,:]
        Tserlist.append(series_slice)
        lenght_prev_cycles+=currentNumberFrames
    return mean_image, Tserlist