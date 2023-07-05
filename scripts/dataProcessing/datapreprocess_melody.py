#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocess functions for melody data

DTW reference: 
    Toni Giorgino (2009). Computing and Visualizing Dynamic Time Warping 
    Alignments in R: The dtw Package. Journal of Statistical Software, 31(7), 
    1-24, doi:10.18637/jss.v031.i07.
    
    pip install dtw-python 1.1.12

@author: barisbozkurt
"""

import os, sys
import copy
import numpy as np
from matplotlib import pyplot as plt
import pickle
import soundfile as sf
import librosa
import crepe
import time
import resampy
import dtw # dtw-python 1.1.12
from scipy.signal import medfilt
from scipy.stats import mode
import pandas as pd

sys.path.append('../') # add root folder to be able to include constants.py
import constants


class Recording(object):

    def __init__(self, file_path, grade, create_figures=False):
        
        self.file_path = file_path.split('/')[-1]
        self.questionID = '_'.join(self.file_path.split('_')[:2])
        self.grade = grade
        self.is_student_performance = '_per' in self.file_path
        # Run pitch estimation or load pitch from file
        self._pitch_estimation(file_path)
        # Run chroma computation or load chroma from file
        self._chroma_computation(file_path)
        # Run f0image computation or load f0image from file
        # self._create_f0img(file_path)
        
    def _pitch_estimation(self, file_path):
        # Loading pitch series if .f0.npy file exists, if not run estimation
        f0Track_file = file_path.replace('.wav','.f0.npy')
        f0Track_file_cent = file_path.replace('.wav','.f0_cent.npy')
        self.freqHopLen_sec = 0.01
        if os.path.exists(f0Track_file):
            f0_series_hz = np.load(f0Track_file).astype('float32')
            # print(f"{f0Track_file} exists, read f0 from file")
        else:
            audio_sig, fs = sf.read(file_path)
            # drop use of crepe due to low computational speed
            time, f0_series_hz, confidence, activation = crepe.predict(audio_sig, fs, viterbi=True)
            self.freqHopLen_sec = time[1] - time[0]
            f0_series_hz[confidence < constants.CREPE_F0_CONFIDENCE_LIMIT] = 0
            
            # drop use of pyin due to low precision for piano sounds
            # librosa_hop_length = int(fs*self.freqHopLen_sec) # 10 msec window length
            # librosa_win_length = librosa_hop_length * 4
            
            # f0, voiced_flag, voiced_probs = librosa.pyin(audio_sig,sr=fs, win_length=librosa_win_length, hop_length=librosa_hop_length, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            # f0[np.isnan(f0)] = 0
            
            # f0_series_hz = f0
            
            # # Drop all low frequency values from the series
            # f0_series_hz = f0_series_hz[f0_series_hz > constants.MINIMUM_FREQUENCY]
            # apply post-filtering to f0-series
            f0_series_hz = medfilt(f0_series_hz, kernel_size=constants.MED_FILT_KERNEL_SIZE_F0)
            f0_series_hz = f0_series_hz.astype('float32')
            np.save(f0Track_file, f0_series_hz) # saving series to .npy file
            # np.savetxt(f0Track_file.replace('.npy','.txt'), f0_series_hz, fmt='%4.2f') # saving series to text file
        
        if os.path.exists(f0Track_file_cent):
            self.f0_series_cent = np.load(f0Track_file_cent).astype('float32')
        else:
            tonic = constants.CENT_REF55HZ            
            self.f0_series_cent = hz2cents_array(f0_series_hz, tonic)
            self.f0_series_cent = self.f0_series_cent.astype('float32')
            np.save(f0Track_file_cent, self.f0_series_cent) # saving series in cents to .npy file
    
    def _chroma_computation(self, file_path):
        '''
        Computes chroma for a given audio file

        Parameters
        ----------
        wav_file : str
            Path to audio file.
        n_fft : int, optional
            Number fo fft points. The default is 2048.
        n_frames : int, optional
            Number of frames for chorma computation. The default is 128.
        version : str, optional
            Version of chroma computation. The default is 'STFT'.
            for other options see https://github.com/meinardmueller/libfmp/blob/master/libfmp/c5/c5s2_chord_rec_template.py

        Returns
        -------
        chroma : numpy arrad
            Chroma feature (dimension: 12, n_frames).

        '''
        # Fetaure dimensions
        n_fft = constants.CHROMA_NFFT
        n_frames = constants.CHROMA_NFRAMES
        
        version = constants.CHROMA_TYPE
        
        # Loading chroma if .chroma.npy file exists, if not run computation
        chroma_file = file_path.replace('.wav','.chroma.npy')

        if os.path.exists(chroma_file):
            self.chroma = np.load(chroma_file).astype('uint8')
        else:        
            x, Fs = sf.read(file_path)
            num_samples_x = x.shape[0]
            hop_length = int((num_samples_x - n_fft) / n_frames)
            if version == 'STFT':
                # Compute chroma features with STFT
                x_stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, pad_mode='constant', center=True)
                x_stft = np.abs(x_stft) ** 2
                chroma = librosa.feature.chroma_stft(S=x_stft, sr=Fs, tuning=0, norm=None, hop_length=hop_length, n_fft=n_fft, n_chroma=constants.CHROMA_NOTES_PER_OCTAVE)
                chroma = librosa.power_to_db(chroma)
                # Normalize and convert to uint8 due to size considerations
                chroma = np.round((chroma - chroma.min()) / (chroma.max() - chroma.min()) * 255).astype('uint8')
                # Drop extra frames
                n_extra_frames = chroma.shape[1] - n_frames
                self.chroma = chroma[:,int(n_extra_frames//2):int(n_extra_frames//2)+n_frames]
                
                # saving feature to .npy file
                np.save(chroma_file, self.chroma) 
            else:
                print('Unknown chroma version!', version)
                self.chroma = np.nan


    def _create_f0img(self, file_path):
        
        f0img_file = file_path.replace('.wav','.f0img.npy')
        
        if not os.path.exists(f0img_file):
            f0_series_cent = np.load(file_path.replace('.wav','.f0_cent.npy'))
            
            img_shape = (constants.F0IMG_FREQ_BINS, constants.F0IMG_NFRAMES)
            f0img = np.zeros(img_shape).astype('uint8')
            threshold_cents = 10 # manually set threshold simply to remove silence parts at both ends
        
            f0_series_cent = strip_series(f0_series_cent, threshold_cents)
            if not self.is_student_performance:
                f0_series_cent = quantize_ref_f0_cent_series(f0_series_cent)
            
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
            self.f0img = f0img
        else:
            self.f0img = np.load(f0img_file)
        pass

def quantize_ref_f0_cent_series(f0_series_cent, step=100):
    '''Quantizes f0 series to a grid with step size in cents
    Applied to reference recording f0 series only
    Additional step of single value in transition removal also applied
    '''
    non_zero_inds = f0_series_cent > 0
    mode_val = mode(f0_series_cent[non_zero_inds].astype('int64')).mode[0]
    f0_series_cent_d = f0_series_cent.copy()
    f0_series_cent_d[non_zero_inds] = np.round((f0_series_cent[non_zero_inds]-mode_val)/step)*step+mode_val
    # filter-out single values in transition bands via assigning them to prev
    diff_sig = np.abs(np.diff(f0_series_cent_d))
    decision = (diff_sig[:-1] > step/2) & (diff_sig[1:] > step/2)
    for i in np.nonzero(decision)[0]:
        if i < f0_series_cent_d.size-1:
            f0_series_cent_d[i+1] = f0_series_cent_d[i]
    # plt.plot(f0_series_cent,'b');plt.plot(f0_series_cent,'r.');
    # plt.figure()
    # plt.plot(f0_series_cent_d,'b');plt.plot(f0_series_cent_d,'r.');
    return f0_series_cent_d

def strip_series(series,threshold=10):
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

def hz2cents_array(array, tonic):
    # set all values below tonic such that resulting cent value woud be zero
    #  this assumes the tonic choosen is a very low value like the freq. of C-1
    array[array <= tonic] = tonic 
    return constants.CENTS_PER_OCTAVE * np.log2(array / float(tonic))
        
def visualize_features(rec_object):
    '''
    Plots features for a Recording object
    '''
    
    plt.figure()
    plt.plot(rec_object.f0_series_cent)
    plt.figure()
    plt.imshow(rec_object.f0img)
    plt.figure()
    plt.imshow(rec_object.chroma[::-1])


def create_recording_objects(target_rec_objects_file, audio_files_folder, annotations_file):
    
    num_samples = 0
    # Data structure to be saved: 
    #    exerciseID -> ref-per Recording object lists
    mel_data = dict()
    start_time = time.time()
    print('Audio feature extraction, ... this is time consuming if feat-files do not already exist')
    audio_not_found_counter = 0
    with open(annotations_file) as list_file:
        for line in list_file:
            file = line.split()[0].strip()
            pat_name = '_'.join(file.split('_')[:2])
            grade = int(line.strip().split('Grade:')[-1])
            if pat_name not in mel_data:
                mel_data[pat_name] = {}
                mel_data[pat_name]['ref'] = []
                mel_data[pat_name]['per'] = []
            
            audio_file_path = os.path.join(audio_files_folder, file)
            if os.path.exists(audio_file_path):
                # Main feature extraction takes place in the next line where Recording object is created
                rec_object = Recording(audio_file_path, grade)
                if rec_object.is_student_performance:
                    mel_data[pat_name]['per'].append(rec_object)
                else:
                    mel_data[pat_name]['ref'].append(rec_object)
                
                num_samples += 1
                if num_samples % 1000 == 0:
                    print('Num files processed:', num_samples)
            # else part Disabled for debugging with low number of files
            else:
                print('Audio file not available:', audio_file_path)
                audio_not_found_counter += 1
                if audio_not_found_counter > 3:
                    print('!!Audio files could not be found')
                    print('Check your data folders')
    stop_time = time.time()
    print('Duration for feature extraction: ', (stop_time - start_time)/60, 'minutes')
    print('Total number of recordings analyzed:', num_samples)
    
    with open(target_rec_objects_file, 'wb') as handle:
            pickle.dump(mel_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print('Data saved to file:', target_rec_objects_file)
    return True

def main():
    # Data file to be created 
    target_rec_objects_file = '../../data/melody/melody-data.pickle'
    audio_files_folder = '../../data/wav/melWav/'
    annotations_file = 'all_annots_2015_2016_mel.txt'
    
    succeed = create_recording_objects(target_rec_objects_file, audio_files_folder, annotations_file);
    
    if not succeed:
        print('Could not create Recording objects')  
        
#---------    
if __name__ == "__main__":
    main()