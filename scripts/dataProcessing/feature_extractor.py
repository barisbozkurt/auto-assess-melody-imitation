#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone feature extractor utils collection

@author: barisbozkurt
"""

import os
import copy
import sys
import numpy as np
from matplotlib import pyplot as plt
# import pickle
import soundfile as sf
import librosa
import crepe
import resampy
# import time
# import dtw # dtw-python 1.1.12
import constants
from scipy.signal import medfilt

def comput_save_chroma(file_path, audio_sig, fs):
    '''
    Computes chroma for a given audio file

    Parameters
    ----------
    wav_file : str
        Path to audio file.
    audio_sig : ndarray
        Audio signal.
    fs : int
        Sampling frequency.
    
    Returns
    -------
    None, saves output to file
    '''
    # Fetaure dimensions
    n_fft = constants.CHROMA_NFFT
    n_frames = constants.CHROMA_NFRAMES
    
    version = constants.CHROMA_TYPE
    
    # Loading chroma if .chroma.npy file exists, if not run computation
    chroma_file = file_path.replace('.wav','.chroma.npy')

    if not os.path.exists(chroma_file):
        num_samples_x = audio_sig.shape[0]
        hop_length = int((num_samples_x - n_fft) / n_frames)
        if version == 'STFT':
            # Compute chroma features with STFT
            x_stft = librosa.stft(audio_sig, n_fft=n_fft, hop_length=hop_length, pad_mode='constant', center=True)
            x_stft = np.abs(x_stft) ** 2
            chroma = librosa.feature.chroma_stft(S=x_stft, sr=fs, tuning=0, norm=None, hop_length=hop_length, n_fft=n_fft)
            chroma = librosa.power_to_db(chroma)
            # Normalize and convert to uint8 due to size considerations
            chroma = np.round((chroma - chroma.min()) / (chroma.max() - chroma.min()) * 255).astype('uint8')
            # Drop extra frames
            n_extra_frames = chroma.shape[1] - n_frames
            chroma = chroma[:,int(n_extra_frames//2):int(n_extra_frames//2)+n_frames]
            
            # saving feature to .npy file
            np.save(chroma_file, chroma) 
        else:
            print('Unknown chroma version!', version)
            chroma = np.nan

def hz2cents_array(f0_array_hz, tonic):
    '''
    Hz to cent converter

    Parameters
    ----------
    f0_array_hz : ndarray
        Array containing f0 values in Hz.
    tonic : float/int
        Reference freq. in Hz.

    Returns
    -------
    NDARRAY
        Array containing f0 values in cents.
    '''
    # set all values below tonic such that resulting cent value woud be zero
    #  this assumes the tonic choosen is a very low value like the freq. of C-1
    f0_array_hz[f0_array_hz <= tonic] = tonic 
    return constants.CENTS_PER_OCTAVE * np.log2(f0_array_hz / float(tonic))

def cents2Hz_array(f0_array_cents):
    '''
    Cents to Hz converter

    Inverse operation for hz2cents_array
    '''
    tonic = constants.CENT_REF55HZ
    f0_array_Hz = np.power(2, f0_array_cents / constants.CENTS_PER_OCTAVE) * tonic
    f0_array_Hz[f0_array_Hz == tonic] = 0
    return  f0_array_Hz

def estimate_save_pitch(file_path, audio_sig, fs):
    '''
    Estimates pitch using Crepe
        Creates two versions of the series and saves in separate files:
            f0 series in Hz saved to *.f0.npy
            f0 series in cent saved to *.f0_cent.npy

    Parameters
    ----------
    file_path : str
        Path to audio file.
    audio_sig : ndarray
        Audio signal.
    fs : int
        Sampling frequency.

    Returns
    -------
    None, saves output to file

    '''
    
    # Loading pitch series if .f0.npy file exists, if not run estimation
    f0Track_file = file_path.replace('.wav','.f0.npy')
    f0Track_file_cent = file_path.replace('.wav','.f0_cent.npy')
    
    if not os.path.exists(f0Track_file):
        
        time, f0_series_hz, confidence, activation = crepe.predict(audio_sig, fs, viterbi=True)
        f0_series_hz[confidence < constants.CREPE_F0_CONFIDENCE_LIMIT] = 0
        f0_series_hz = medfilt(f0_series_hz, kernel_size=constants.MED_FILT_KERNEL_SIZE_F0)
        f0_series_hz = f0_series_hz.astype('float32')
        np.save(f0Track_file, f0_series_hz) # saving series in Hz to .npy file
        # np.savetxt(f0Track_file.replace('.npy','.txt'), f0_series_hz, fmt='%4.2f') # saving series to text file
        
        # Convert to cents using 55Hz as reference and saving to a file
        f0_series_cent = hz2cents_array(f0_series_hz, constants.CENT_REF55HZ)
        f0_series_cent = f0_series_cent.astype('float32')
        np.save(f0Track_file_cent, f0_series_cent) # saving series in cents to .npy file
    
    # if f0Track_file exists but not the f0Track_file_cent
    elif not os.path.exists(f0Track_file_cent):
        f0_series_hz = np.load(f0Track_file)
        # Convert to cents using 55Hz as reference and saving to a file
        f0_series_cent = hz2cents_array(f0_series_hz, constants.CENT_REF55HZ)
        f0_series_cent = f0_series_cent.astype('float32')
        np.save(f0Track_file_cent, f0_series_cent) # saving series in cents to .npy file
        

def strip_series(series,threshold):
    '''Strips a numerical series as string stripping:  
    The series is cropped at both ends to exclude values lower than the threshold
    '''
    bool_ind = series < threshold
    low_limit_ind = 0 # crop start index
    high_limit_ind = series.size-1 # crop end index
    for i in range(series.size):
        if not bool_ind[i]:
            low_limit_ind = i
            break
    for i in range(series.size-1,0,-1):
        if not bool_ind[i]:
            high_limit_ind = i
            break
    return series[low_limit_ind:high_limit_ind+1]

def create_f0img(file_path):
    
    f0img_file = file_path.replace('.wav','.f0img.npy')
    
    if not os.path.exists(f0img_file):
        f0_series_cent = np.load(file_path.replace('.wav','.f0_cent.npy'))
        
        img_shape = (constants.F0IMG_FREQ_BINS, constants.F0IMG_NFRAMES)
        f0img = np.zeros(img_shape).astype('uint8')
        threshold_cents = 10 # manually set threshold simply to remove silence parts at both ends
    
        f0_series_cent = strip_series(f0_series_cent, threshold_cents)
        
        # Adding f0-trace in f0img. f0 series is warped in an octave first
        #   However, computing mod-1200 introduces discontinuities, 
        #   copies at upper and lower octaves are created to compansate that
        #   So, three traces added: warped versions of orignal, shifted to 
        #   one higher octave and shifted to one lower octave
        f0_series_cent_org = f0_series_cent.copy()
        for val_2_add in [0, constants.CENTS_PER_OCTAVE, -constants.CENTS_PER_OCTAVE]:
            f0_series_cent = f0_series_cent_org + val_2_add
            # warp frequency values in two octaves
            f0_series_cent = np.mod(f0_series_cent, 2*constants.CENTS_PER_OCTAVE)
            
            # Convert values in cents to indexes in f0img (freq. dim.)
            f0_series_inds = (constants.F0IMG_FREQ_BINS * f0_series_cent/(2*constants.CENTS_PER_OCTAVE)).astype('int16')
            
            # Set values in f0img
            #  If length is too-small, upsample the series
            if f0_series_inds.size < constants.F0IMG_NFRAMES:
                f0_series_inds = resampy.resample(f0_series_inds,f0_series_inds.size, constants.F0IMG_NFRAMES*2, axis=-1)
            step = f0_series_inds.size/constants.F0IMG_NFRAMES
            for k in range(constants.F0IMG_NFRAMES):
                frame_start = int(k*step); frame_stop = int((k+1)*step)
                mean_val_frame = int(np.mean(f0_series_inds[frame_start:frame_stop]))
                if np.isnan(mean_val_frame):
                    mean_val_frame = 0
                # to ensure the same shape in plots during debugging, invert the index
                freq_index = min(max(0, constants.F0IMG_FREQ_BINS - mean_val_frame), constants.F0IMG_FREQ_BINS-1)
                f0img[freq_index, k] = 255
        
        f0img = f0img.astype('uint8')
        np.save(f0img_file, f0img) # saving series in cents to .npy file
        

def extract_features_for_audio(file_path):
    '''
    Runs feature extraction for one audio file

    Parameters
    ----------
    file_path : str
        Path to audio file.

    Returns
    -------
    None. Writes outputs to files

    '''
    # Read audio signal
    audio_sig, fs = sf.read(file_path)
    
    # Run f0-extraction
    estimate_save_pitch(file_path, audio_sig, fs)
    
    # Run f0-image computation
    create_f0img(file_path)
    
    # Run chromagram computation
    comput_save_chroma(file_path, audio_sig, fs)
    
def visualize_features(file_path):
    '''
    Plots features for an audio file

    Parameters
    ----------
    file_path : str
        Path to audio file.

    Returns
    -------
    None.

    '''
    
    f0_series_cent = np.load(file_path.replace('.wav','.f0_cent.npy'))
    chroma = np.load(file_path.replace('.wav','.chroma.npy'))
    f0img = np.load(file_path.replace('.wav','.f0img.npy'))
    plt.figure()
    plt.plot(f0_series_cent)
    plt.figure()
    plt.imshow(f0img)
    plt.figure()
    plt.imshow(chroma)

def main():
    # Data file to be created 
    audio_files_folder = '../../data/wav/melWav/'

    # Extracting feature for each audio file
    num_files_processed = 0
    for root, dirs, files in os.walk(audio_files_folder):
        for file in files:
            if file.endswith('.wav'): 
                file_path = os.path.join(root, file)
                extract_features_for_audio(file_path)
                #visualize_features(file_path)
                num_files_processed += 1
                if (num_files_processed % 100) == 0:
                    print('Number of files processed:', num_files_processed)
    
    

    
        
#---------    
if __name__ == "__main__":
    main()