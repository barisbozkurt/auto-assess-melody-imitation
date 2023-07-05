#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:35:45 2023

Reads and resamples f0-series in cents and stores new series 
in *f0_cent_fixL.npy files
    
Target length defined by constants.F0IMG_NFRAMES

@author: barisbozkurt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import resampy
import constants

def resample_f0_cent(source_f0_cent_file, target_f0_cent_file, force_recompute=False):
    '''
    Reads f0-series in cents from source_f0_cent_file, 
    removes silences, resamples series and saves new series in 
    target_f0_cent_file

    Parameters
    ----------
    source_f0_cent_file : str
        Path to original f0-cent file.
    target_f0_cent_file : str
        Path to target/resampled f0-cent file.
    force_recompute : bool
        Flag to control forcing re-computation even if target file exists

    Returns
    -------
    None.

    '''
    
    if not os.path.exists(target_f0_cent_file) or force_recompute:
        # Read file
        f0_cent = np.load(source_f0_cent_file)
        
        # Use the same length as F0IMG for f0-series
        target_len = constants.F0IMG_NFRAMES
        
        # drop silences
        f0_cent = f0_cent[f0_cent > 0]
        
        # Resampling adds artifacts at boundaries, so add extra samples at both ends
        # resample and drop samples from boundaries
        num_extra = 4 # number of extra points on both sides after resampling  
        num_samples2add = int(f0_cent.size * 4/constants.F0IMG_NFRAMES)
        f0_cent = np.concatenate((np.ones((num_samples2add,))*f0_cent[0], f0_cent, np.ones((num_samples2add,))*f0_cent[-1]))
    
        # Resample 
        f0_cent = resampy.resample(f0_cent, f0_cent.size, target_len + 2*num_extra)
        f0_cent = f0_cent[num_extra:num_extra+constants.F0IMG_NFRAMES]
            
        # Write to file
        np.save(target_f0_cent_file, f0_cent) 
    
def plot_after_resamp(source_file_path, target_file_path):
    '''Plotting function for debugging purposes:
        Plots the original and resampled versions of the series'''
    f0_org = np.load(source_file_path)
    f0_fixL = np.load(target_file_path)
    plt.figure()
    plt.plot(np.linspace(0,1,f0_org.size), f0_org, 'b', label='original')
    plt.plot(np.linspace(0,1,f0_fixL.size), f0_fixL, 'r', label='resampled')
    plt.legend()

def resample_f0files_inFolder(dba_folder):
    '''Runs resampling for all *f0_cent.npy files in folder dba_folder'''
    for root, dirs, files in os.walk(dba_folder):
        for file in files:
            if file.endswith('f0_cent.npy'): # annotation files starts with 'report'
                source_file_path = os.path.join(root, file)
                target_file_path = source_file_path.replace('f0_cent.npy', 'f0_cent_fixL.npy')
                resample_f0_cent(source_file_path, target_file_path)
                # plot_after_resamp(source_file_path, target_file_path)    

def main():
    dba_folder = '../../data/wav/melWav/'
    resample_f0files_inFolder(dba_folder)
            
if __name__ == "__main__":
    main()  