#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for creating some sample plots used for debugging and 
creating figures used in the paper

@author: barisbozkurt
"""
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import pickle
import constants
from datapreprocess_melody import Recording, visualize_features

def get_rec_obj(mel_data, file_name, pattern_name, object_type):
    '''Returns the recording object with file_name
    mel_data: collection data read from 'melody-data.pickle'
    object_type: 'ref' or 'per'
    '''
    for obj in mel_data[pattern_name][object_type]:
        if obj.file_path == file_name:
            # print('Object for {} retrieved:'.format(file_name))
            return obj

def cents2Hz_array(f0_array_cents):
    '''
    Cents to Hz converter

    Inverse operation for hz2cents_array
    '''

    tonic = constants.CENT_REF55HZ
    f0_array_Hz = np.power(2, f0_array_cents / constants.CENTS_PER_OCTAVE) * tonic
    f0_array_Hz[f0_array_Hz == tonic] = 0
    return  f0_array_Hz

def plot_f0_on_spectrogram(rec_obj, audio_data_folder, figures_folder, plot_max_f0=600):
    """ 
    file_name: Path without file extension
    """
    figure_file_name = os.path.join(figures_folder,rec_obj.file_path.replace('.wav','_specF0chroma.png'))
    
    if not os.path.exists(figure_file_name):
        # Read audio signal
        audio_sig, fs = sf.read(os.path.join(audio_data_folder, rec_obj.file_path))
        hop_length = int(0.01 * fs)
        win_length = 4 * hop_length
        w = get_window("blackman", win_length)
    
        # Retrieve f0-series in Hz
        f0_series_Hz = cents2Hz_array(rec_obj.f0_series_cent)
    
        fft_N = 4096
        start_inds = np.arange(0, audio_sig.size - win_length, hop_length, dtype=int)
        spctrgrm = np.array([]).reshape(0, int(fft_N / 2))
        for k in range(start_inds.size):
            start_ind = start_inds[k]
            x_w = audio_sig[start_ind : start_ind + win_length] * w
            X = np.fft.fft(x_w, fft_N)
            genlikX = np.abs(X[: int(fft_N / 2)])
            genlikX[genlikX < np.finfo(float).eps] = np.finfo(
                float
            ).eps  # log operasyonundan önce önlem
            genlik_spektrumu = 20 * np.log10(genlikX)
            spctrgrm = np.vstack(
                (spctrgrm, genlik_spektrumu)
            )  # k. sinyal kesitinin spektrumunun eklenmesi
    
        # Plot spectrogram matrix
        time_ax = np.arange(spctrgrm.shape[0]) * hop_length / float(fs)
        freq_ax_Hz = np.arange(spctrgrm.shape[1]) * float(fs) / float(fft_N)
    
        time_ax_f0 = np.linspace(0, time_ax[-1], f0_series_Hz.shape[0])
        
        captionFontSize = 24
        plt.figure(figsize=(18, 20))
        plt.subplot(4,1,1)
        rec_file_name = rec_obj.file_path.replace('.wav','').replace('_pass','').replace('_fail','')
        plt.title(f"{rec_file_name}: Sinyal dalga formu",fontsize=captionFontSize+2)
        time_x_ax = np.arange(audio_sig.size)/fs
        plt.plot(time_x_ax,audio_sig)
        plt.xlim([0, time_x_ax[-1]])
        plt.ylabel("genlik",fontsize=captionFontSize)
        plt.subplot(4,1,2)
        plt.pcolormesh(time_ax, freq_ax_Hz, np.transpose(spctrgrm))
        # plt.plot(time_ax_f0, f0_series_Hz, "r")
        # plt.ylim([0, np.max(f0_series_Hz)*2])
        plt.title("Genlik Spektrogramı (dB)",fontsize=captionFontSize+2)
        plt.ylabel("frekans(Hz)",fontsize=captionFontSize)
        #plt.xlabel("zaman(saniye)")
        plt.ylim([0, np.max(f0_series_Hz)*10])

        plt.subplot(4,1,3)
        plt.title("Genlik Spekt.(dB) ve f0 serisi",fontsize=captionFontSize+2)
        plt.pcolormesh(time_ax, freq_ax_Hz, np.transpose(spctrgrm))
        plt.plot(time_ax_f0, f0_series_Hz, "r")
        plt.ylim([0, np.max(f0_series_Hz)*2])
        plt.ylabel("frekans(Hz)",fontsize=captionFontSize)
        plt.xlabel("zaman(saniye)",fontsize=captionFontSize)

        plt.subplot(4,1,4)
        plt.title('Kromagram',fontsize=captionFontSize+2)
        plt.imshow(rec_obj.chroma[::-1]);
        plt.ylabel("kroma endeksi",fontsize=captionFontSize)
        plt.xlabel('pencere endeksi (n)',fontsize=captionFontSize);
        
        plt.savefig(figure_file_name, dpi=300)
        plt.close()

audio_data_folder = '../../data/wav/melWav/'    
data_folder = '../../data/melody/melody_data4ML_1Asli'
figures_folder = 'sample_figures'
data_file_name = 'melody-data.pickle'
with open(os.path.join(data_folder, data_file_name), 'rb') as handle:
    mel_data_read = pickle.load(handle)

test_chromas = np.load(os.path.join(data_folder, 'test_chroma_X.npy'))
test_f0img = np.load(os.path.join(data_folder, 'test_f0img_X.npy'))
test_alignedF0 = np.load(os.path.join(data_folder, 'test_aligned_f0_X.npy'))

with open(os.path.join(data_folder, 'train_test_file_names.pickle'), 'rb') as handle:
    train_test_files = pickle.load(handle)
(train_comb_files, test_comb_files) = train_test_files

if not os.path.exists(figures_folder):
    os.mkdir(figures_folder)

subfolders = ['1','2','3','4']
for subfolder in subfolders:
    if not os.path.exists(figures_folder+'/'+subfolder):
        os.mkdir(figures_folder+'/'+subfolder)

file_counter_per_grade = {1:0,2:0,3:0,4:0}
max_num_file_per_grade = 3

#for ind in range(5):
for ind in range(len(test_comb_files)):
    if ind % 10 == 0:
        print(ind)
    test_couple = test_comb_files[ind]
    test_ref_file = test_couple.split()[0]
    test_per_file = test_couple.split()[1]
    pattern_name = '_'.join(test_ref_file.split('_')[:2])
    
    ref_rec_obj = get_rec_obj(mel_data_read, test_ref_file, pattern_name, 'ref')
    per_rec_obj = get_rec_obj(mel_data_read, test_per_file, pattern_name, 'per')
    
    grade = per_rec_obj.grade
    # create max_num_file_per_grade figures for each grade, skip the rest 
    if file_counter_per_grade[grade] < max_num_file_per_grade:
        file_counter_per_grade[grade] += 1
    
        # # Visualize recording object features
        # visualize_features(ref_rec_obj)
        # visualize_features(per_rec_obj)
        
        # # Visualize packed features for comparison
        plt.figure()
        plt.subplot(2,1,1)
        per_filename = per_rec_obj.file_path.replace('.wav','').replace('_pass','').replace('_fail','')
        ref_filename = ref_rec_obj.file_path.replace('.wav','').replace('_pass','').replace('_fail','')
        file_name = per_filename + '-' + ref_filename
        plt.plot(ref_rec_obj.f0_series_cent,'b');plt.title('{}, Not:{}'.format(file_name, grade))
        plt.plot(per_rec_obj.f0_series_cent,'r');
        min_y = 0.9 * min(np.min(ref_rec_obj.f0_series_cent[ref_rec_obj.f0_series_cent > 100]),
                    np.min(per_rec_obj.f0_series_cent[per_rec_obj.f0_series_cent > 100]))
        max_y = 1.1 * max(np.max(ref_rec_obj.f0_series_cent),
                    np.max(per_rec_obj.f0_series_cent))
        plt.ylim(min_y, max_y)
        plt.ylabel('f0 (sent)')
            
        plt.subplot(2,1,2)
        plt.plot(test_alignedF0[ind][0],'b', label='referans kayıt');
        plt.plot(test_alignedF0[ind][1],'r', label='öğr. performans kaydı');
        plt.ylim(min_y, max_y)
        plt.xlabel('seri örnek endeksi (n)');plt.ylabel('f0 (sent)')
        plt.legend()
        plt.savefig(os.path.join(figures_folder+'/'+str(grade), file_name+'.png'), dpi=300)
        plt.close()
        
        plot_f0_on_spectrogram(ref_rec_obj, audio_data_folder, figures_folder)
        plot_f0_on_spectrogram(per_rec_obj, audio_data_folder, figures_folder)
        
        # figure_file_name = os.path.join(figures_folder,ref_rec_obj.file_path.replace('.wav','')+'_f0img.png')
        # if not os.path.exists(figure_file_name):
        #     plt.figure()
        #     plt.imshow(ref_rec_obj.f0img);plt.title('ref')
        #     plt.savefig(figure_file_name, dpi=300)
        #     plt.close()
        
        # figure_file_name = os.path.join(figures_folder,per_rec_obj.file_path.replace('.wav','')+'_f0img.png')
        # if not os.path.exists(figure_file_name):
        #     plt.figure()
        #     plt.imshow(per_rec_obj.f0img);plt.title('per')
        #     plt.savefig(figure_file_name, dpi=300)
        #     plt.close()
    
        
        # figure_file_name = os.path.join(figures_folder,ref_rec_obj.file_path.replace('.wav','')+'_chroma.png')
        # if not os.path.exists(figure_file_name):
        #     plt.figure()
        #     plt.imshow(ref_rec_obj.chroma[::-1]);plt.title('ref') 
        #     plt.savefig(figure_file_name, dpi=300)
        #     plt.close()
        
        # figure_file_name = os.path.join(figures_folder,per_rec_obj.file_path.replace('.wav','')+'_chroma.png')
        # if not os.path.exists(figure_file_name):
        #     plt.figure()
        #     plt.imshow(per_rec_obj.chroma[::-1]);plt.title('per')
        #     plt.savefig(figure_file_name, dpi=300)
        #     plt.close()
    if file_counter_per_grade == {1:max_num_file_per_grade,
                                  2:max_num_file_per_grade,
                                  3:max_num_file_per_grade,
                                  4:max_num_file_per_grade}:
        break
   




