'''
Reads Recording objects from 'melody-data.pickle'
and groups features to create machine learning data

DTW reference: 
    Toni Giorgino (2009). Computing and Visualizing Dynamic Time Warping 
    Alignments in R: The dtw Package. Journal of Statistical Software, 31(7), 
    1-24, doi:10.18637/jss.v031.i07.
    
    pip install dtw-python 1.1.12
    
DTW on chroma reference:

    Implementation resource: https://meinardmueller.github.io/libfmp/build/html/index.html
    Meinard Müller and Frank Zalkow. libfmp: A Python Package for 
    Fundamentals of Music Processing. Journal of Open Source Software 
    (JOSS), 6(63), 2021.
    
@author: barisbozkurt
'''

import dtw # dtw-python 1.1.12

import numpy as np
import os
# import crepe
import constants
from scipy.signal import medfilt
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datapreprocess_melody import Recording, strip_series, quantize_ref_f0_cent_series
from convert_annotations import write_list_to_file
import resampy
import libfmp.c3


def compute_distance_features_f0(ref_recording, std_recording, create_figures_ON=False):
    '''
    Alignes ref_recording and std_recording (f0 series in cents) and computes 
    distances features.
    
    The features to be computed are listed in constants.py (feature_names)

    Parameters
    ----------
    ref_recording : Recording object
        Reference recording object.
    std_recording : Recording object
        Student performance recording object.
    create_figures_ON : bool, optional
        Flag to control figure creation. The default is False.

    Returns
    -------
    f0_dist_features : dict
        Distance features computed.

    '''
    ref_f0_series_cent = ref_recording.f0_series_cent.copy()
    ref_f0_series_cent = ref_f0_series_cent[ref_f0_series_cent > 0] # silences are irrelevant in comparison
    ref_f0_series_cent = quantize_ref_f0_cent_series(ref_f0_series_cent)
    std_f0_series_cent = std_recording.f0_series_cent.copy()
    std_f0_series_cent = std_f0_series_cent[std_f0_series_cent > 0] # silences are irrelevant in comparison
    
    # Octave mismatch correction
    diff_mean_series = np.mean(ref_f0_series_cent) - np.mean(std_f0_series_cent)
    if diff_mean_series > constants.CENTS_PER_OCTAVE/2:
        shift_amount = np.round(diff_mean_series / constants.CENTS_PER_OCTAVE) * constants.CENTS_PER_OCTAVE
        ref_f0_series_cent -= shift_amount
    elif diff_mean_series < -constants.CENTS_PER_OCTAVE/2:
        shift_amount = np.round(-diff_mean_series / constants.CENTS_PER_OCTAVE) * constants.CENTS_PER_OCTAVE
        ref_f0_series_cent += shift_amount
        
    # Resample the long array to match the length of the short since
    #  our task is tempo invariant and we prefer to downsample than upsample 
    #  because upsampling may add high freq components at discontinuities
    if ref_f0_series_cent.size > std_f0_series_cent.size:
        ref_f0_series_cent = resampy.resample(ref_f0_series_cent, ref_f0_series_cent.size, std_f0_series_cent.size)
    elif ref_f0_series_cent.size < std_f0_series_cent.size:
        std_f0_series_cent = resampy.resample(std_f0_series_cent, std_f0_series_cent.size, ref_f0_series_cent.size)

    
    #Perform dtw alignment
    # window_type= 'itakura'; window_args={}
    # alignment = dtw.dtw(ref_f0_series_cent, std_f0_series_cent,window_type=window_type, window_args=window_args)
    alignment = dtw.dtw(ref_f0_series_cent, std_f0_series_cent)
    ref_f0_series_cent_aligned = ref_f0_series_cent[alignment.index1]
    ref_recording.f0_series_cent_aligned = ref_f0_series_cent_aligned
    std_f0_series_cent_aligned = std_f0_series_cent[alignment.index2]
    std_recording.f0_series_cent_aligned = std_f0_series_cent_aligned
    
    #Features computed from dtw alignment
    f0_dist_features = {}
    f0_dist_features['f0_dtw_norm_dist'] = alignment.normalizedDistance
    f0_dist_features['f0_dtw_len_mod_ref'] = alignment.index1.size / ref_f0_series_cent.size
    f0_dist_features['f0_dtw_len_mod_std'] = alignment.index2.size / std_f0_series_cent.size
    #Features computed from diff signal
    diff_series = np.abs(std_f0_series_cent_aligned - ref_f0_series_cent_aligned)

    f0_dist_features['f0_mean_diff'] = np.mean(diff_series)
    f0_dist_features['f0_std_diff'] = np.std(diff_series)
    f0_dist_features['f0_last_10perc_mean_diff'] = np.mean(diff_series[-diff_series.size//10:])
    
    diff_hist = np.histogram(diff_series, constants.diff_hist_bins)[0]
    diff_hist = diff_hist / np.sum(diff_hist)
    for i, val in enumerate(diff_hist):
        f0_dist_features['f0_diff_hist'+str(i)] = val
    
    if create_figures_ON:
        plt.plot(ref_recording.f0_series_cent_aligned,'b', label='ref_aligned')
        plt.plot(std_recording.f0_series_cent_aligned,'r', label='std_aligned')
        max_val_f0 = np.max(ref_recording.f0_series_cent_aligned)
        mean_val_f0 = np.mean(ref_recording.f0_series_cent_aligned)
        plt.text(0, max_val_f0, 'f0_dtw_norm_dist:{:.2f}'.format(f0_dist_features['f0_dtw_norm_dist'][0]))
        plt.text(0, (max_val_f0+mean_val_f0)/2, 'f0_mean_diff:    {:.2f}'.format(f0_dist_features['f0_mean_diff'][0]))
        plt.text(0, mean_val_f0, 'f0_std_diff:     {:.2f}'.format(f0_dist_features['f0_std_diff'][0]))
        
        if not os.path.exists('figures'):
            os.mkdir('figures')
        ref_file = ref_recording.file_path.split('/')[-1].replace('.wav','')
        out_fig_file = os.path.join('figures',std_recording.file_path.split('/')[-1].replace('.wav','')+'_'+ref_file+'.align.png')

        plt.savefig(out_fig_file)
        plt.close()
    
    return f0_dist_features

def compute_distance_features_chroma(ref_recording, std_recording):
    ref_chroma = ref_recording.chroma.copy()
    std_chroma = std_recording.chroma.copy()

    # Application of dtw on chroma
    # Implementation resource: https://meinardmueller.github.io/libfmp/build/html/index.html
    # Meinard Müller and Frank Zalkow. libfmp: A Python Package for 
    # Fundamentals of Music Processing. Journal of Open Source Software 
    # (JOSS), 6(63), 2021.

    cost_matrix = libfmp.c3.compute_cost_matrix(ref_chroma, std_chroma) # default distance metric: euclidean
    acc_cost_matrix = libfmp.c3.compute_accumulated_cost_matrix(cost_matrix)
    optimal_warping_path = libfmp.c3.compute_optimal_warping_path(acc_cost_matrix)
    
    ref_chroma_aligned = ref_chroma[:, optimal_warping_path[:,0]]
    std_chroma_aligned = std_chroma[:, optimal_warping_path[:,1]]
    
    #Features computed from dtw alignment
    chroma_dist_features = {}
    chroma_dist_features['chrm_dtw_norm_dist'] = acc_cost_matrix[-1, -1]
    chroma_dist_features['chrm_dtw_len_mod_ref'] = len(optimal_warping_path[:,0]) / ref_chroma.size
    chroma_dist_features['chrm_dtw_len_mod_std'] = len(optimal_warping_path[:,1]) / ref_chroma.size
    #Features computed from diff signal
    diff_series = np.abs(ref_chroma_aligned - std_chroma_aligned)
    #Convert to ratio of change wrt reference chroma in percentage
    diff_series = 100*diff_series.sum(axis=0)/ref_chroma_aligned.sum(axis=0)    
    
    # reset_diff_for_low_f0(diff_series, std_f0_series_cent_aligned, ref_f0_series_cent_aligned)

    chroma_dist_features['chrm_mean_diff'] = np.mean(diff_series)
    chroma_dist_features['chrm_std_diff'] = np.std(diff_series)
    chroma_dist_features['chrm_last_10perc_mean_diff'] = np.mean(diff_series[-diff_series.size//10:])
    
    diff_hist = np.histogram(diff_series, constants.diff_hist_bins)[0]
    diff_hist = diff_hist / np.sum(diff_hist)
    for i, val in enumerate(diff_hist):
        chroma_dist_features['chrm_diff_hist'+str(i)] = val
    
    return chroma_dist_features

def read_test_combinations_in_list(file_path):
    '''
    Retrieve file couples list from file

    Parameters
    ----------
    file_path : str
        Path to text file containing ref-per file couples .

    Returns
    -------
    test_combinations : list
        Couples.
    ref_files : list
        Reference files.
    per_files : TYPE
        Student performance files.
    '''
    test_combinations = []
    ref_files = []
    per_files = []
    with open(file_path) as file:
        for line in file:
            if len(line) > 2:
                test_combinations.append(line.strip().replace('\t',' '))
                ref_file = line.strip().split()[0]
                per_file = line.strip().split()[-1]
                if ref_file not in ref_files:
                    ref_files.append(ref_file)
                if per_file not in per_files:
                    per_files.append(per_file)
    return test_combinations, ref_files, per_files

def add_column_names_to_csv(file_pointer, feature_names_list):
    '''Adds column names to csv file (the first row of the csv file)'''
    file_pointer.write('Ref_file,Per_file,')
    for feature_names in feature_names_list:
        for dist_name in feature_names:
            if dist_name != feature_names[-1]:
                file_pointer.write('{},'.format(dist_name))
            else:
                file_pointer.write('{}'.format(dist_name))
    file_pointer.write('\n')
    
def add_line_to_csv(file_pointer, ref_file_path, per_file_path, dists_list, grade):
    '''Adds a line (sample) to the csv file'''
    file_pointer.write(ref_file_path.replace('.wav','') + ',' + per_file_path.replace('.wav','') + ',')
    for dists in dists_list:
        for dist in dists:
            file_pointer.write('{:.4f},'.format(dist))
    file_pointer.write('{}\n'.format(grade)) 


def compute_distance_features(ref_rec, per_rec):
    f0_dist_features = compute_distance_features_f0(ref_rec, per_rec)
    f0_dists = [f0_dist_features[fname] for fname in constants.feature_names if 'f0' in fname]
    
    chroma_dist_features = compute_distance_features_chroma(ref_rec, per_rec)
    chroma_dists = [chroma_dist_features[fname] for fname in constants.feature_names if 'chrm' in fname]
    
    return chroma_dists, f0_dists

def add_data_sample(ref_rec, per_rec, X_chroma, X_f0img, X_alignedF0, y, comb_files, dataFrame_f_wr, ind):
    '''Adds sample to ML data
    X: numpy vector containing chroma couples, f0img couples, aligned f0 couples
    y: grades for the couples
    comb_files: list of file couples
    dataFrame_f_wr: pointer to csv file containing ML data (a line is written to the file as a sample)
    ind: index in couples collection
    '''
    # Compute distances after aligning f0_cent series.
    #  this function also adds new data members to the objects: f0_series_cent_aligned
    chroma_dists, f0_dists = compute_distance_features(ref_rec, per_rec)
    add_line_to_csv(dataFrame_f_wr, ref_rec.file_path, per_rec.file_path, 
                    [f0_dists, chroma_dists], per_rec.grade)
    
    X_chroma[ind, 0] = ref_rec.chroma.astype('uint8')
    X_chroma[ind, 1] = per_rec.chroma.astype('uint8')
    # X_f0img[ind, 0] = ref_rec.f0img.astype('uint8') # this feature is disabled, was not useful in tests
    # X_f0img[ind, 1] = per_rec.f0img.astype('uint8')
    
    # resample aligned f0 before storage
    ref_f0_aligned = resampy.resample(ref_rec.f0_series_cent_aligned, ref_rec.f0_series_cent_aligned.shape[0], constants.ALIGNED_F0_LEN)
    ref_f0_aligned[ref_f0_aligned < 0] = 0
    if ref_f0_aligned.size < constants.ALIGNED_F0_LEN:
        ref_f0_aligned = np.concatenate((ref_f0_aligned, np.zeros((constants.ALIGNED_F0_LEN-ref_f0_aligned.size,))))

    per_f0_aligned = resampy.resample(per_rec.f0_series_cent_aligned, per_rec.f0_series_cent_aligned.shape[0], constants.ALIGNED_F0_LEN)
    per_f0_aligned[per_f0_aligned < 0] = 0
    if per_f0_aligned.size < constants.ALIGNED_F0_LEN:
        per_f0_aligned = np.concatenate((per_f0_aligned, np.zeros((constants.ALIGNED_F0_LEN-per_f0_aligned.size,))))
    
    X_alignedF0[ind, 0] = ref_f0_aligned.astype('uint16')
    X_alignedF0[ind, 1] = per_f0_aligned.astype('uint16')
    
    y[ind] = per_rec.grade
    comb_files.append(ref_rec.file_path + ' ' + per_rec.file_path)

    
def prepare_ML_data(data_file_name, train_couples_file, test_couples_file, exclude_ref_list_file):
    '''Main function to: 
        - read the Recording objects (from the pickle file), 
        - read the file list files, 
        - group ref-per pairs of data to:
            - compute pair-wise f0-series distances and save distance features to csv files
            - create test and train files containing input-data-pairs and outputs(grades)            
    '''
    #reading pickle
    with open(data_file_name, 'rb') as handle:
        mel_data_read = pickle.load(handle)
        
    # Read file lists for test and train 
    train_combinations, train_ref_files, train_per_files = read_test_combinations_in_list(train_couples_file)
    test_combinations, test_ref_files, test_per_files = read_test_combinations_in_list(test_couples_file)
    
    # (Over)writing lists to text files (test_couples.txt, train_couples.txt may have been overwritten)
    #  See line 182-188 of dataProcessPipeline.py
    write_list_to_file(test_per_files, 'test_per_files.txt')
    write_list_to_file(train_per_files, 'train_per_files.txt')
    write_list_to_file(test_ref_files, 'test_ref_files.txt')
    write_list_to_file(train_ref_files, 'train_ref_files.txt')  
    
    # Read reference files excluded because they received a grade lower than 4
    if os.path.exists(exclude_ref_list_file):
        ref_files_2_exclude = []
        with open(exclude_ref_list_file) as file:
            for line in file:
                if len(line) > 2:
                    ref_files_2_exclude.append(line.strip().split()[0])
        
        # Sanity check: check if excluded ref files exist in target files
        for train_ref_file in train_ref_files:
            if train_ref_file in ref_files_2_exclude:
                print('!!!Error: reference file to be excluded exists in the train set:', train_ref_file)
        
        for test_ref_file in test_ref_files:
            if test_ref_file in ref_files_2_exclude:
                print('!!!Error: reference file to be excluded exists in the test set:', test_ref_file)
    
    
    # Creating data files comsumed by the base-line system: .csv files containing 
    # distances as features and the grade at the last column
    test_dataFrame_f = 'testData.csv'
    test_dataFrame_f_wr = open(test_dataFrame_f, 'w')
    # Add column names to data frame, last column is reserved for grade
    add_column_names_to_csv(test_dataFrame_f_wr, [constants.feature_names+['grade']])
    
    train_dataFrame_f = 'trainData.csv'
    train_dataFrame_f_wr = open(train_dataFrame_f, 'w')
    # Add column names to data frame
    add_column_names_to_csv(train_dataFrame_f_wr, [constants.feature_names+['grade']])
    
    # Also create data structures to collect chroma features
    #  each sample is a chroma couple (ref, per)
    num_test_samples = len(test_combinations)
    X_test_chroma = np.zeros((num_test_samples, 2, constants.CHROMA_NOTES_PER_OCTAVE, constants.CHROMA_NFRAMES)).astype('uint8')
    # X_test_f0img = np.zeros((num_test_samples, 2, constants.F0IMG_FREQ_BINS, constants.F0IMG_NFRAMES)).astype('uint8')
    X_test_f0img = [] # this feature is disabled, was not useful in tests
    X_test_alignedF0 = np.zeros((num_test_samples, 2, constants.ALIGNED_F0_LEN)).astype('uint16')
    y_test = np.zeros((num_test_samples, ))
    
    num_train_samples = len(train_combinations)
    X_train_chroma = np.zeros((num_train_samples, 2, constants.CHROMA_NOTES_PER_OCTAVE, constants.CHROMA_NFRAMES)).astype('uint8')
    # X_train_f0img = np.zeros((num_train_samples, 2, constants.F0IMG_FREQ_BINS, constants.F0IMG_NFRAMES)).astype('uint8')
    X_train_f0img = [] # this feature is disabled, was not useful in tests
    X_train_alignedF0 = np.zeros((num_train_samples, 2, constants.ALIGNED_F0_LEN)).astype('uint16')
    y_train = np.zeros((num_train_samples, ))
    
    print('Number of test and train couples to be processed: ', (num_train_samples + num_test_samples))
    test_ind = 0
    train_ind = 0
    train_comb_files = [] # used for debugging purposes
    test_comb_files = [] # used for debugging purposes
    for exercise in mel_data_read.keys():
        # print('Processing data for exercise', exercise)
        for ref_rec in mel_data_read[exercise]['ref']:
            for per_rec in mel_data_read[exercise]['per']:
                if ((per_rec.file_path in test_per_files) and # to speed up process: compute if combination is in test data or train data
                    (ref_rec.file_path in test_ref_files) or 
                    (per_rec.file_path in train_per_files) and 
                    (ref_rec.file_path in train_ref_files)):
                    
                    
                    # Computing dist features for ref and per 
                    #  and creating a data sample, adding to data collections
                    if (ref_rec.file_path + ' ' + per_rec.file_path) in test_combinations:
                        add_data_sample(ref_rec, per_rec, X_test_chroma, X_test_f0img, X_test_alignedF0, y_test, 
                                        test_comb_files, test_dataFrame_f_wr, test_ind)
                        test_ind += 1
                    elif (ref_rec.file_path + ' ' + per_rec.file_path) in train_combinations:
                        add_data_sample(ref_rec, per_rec, X_train_chroma, X_train_f0img, X_train_alignedF0, y_train, 
                                        train_comb_files, train_dataFrame_f_wr, train_ind)
                        train_ind += 1
                        
                    if (test_ind + train_ind) > 0 and (test_ind + train_ind) % 10000 == 0:
                        print('Number of data couples saved:', (test_ind + train_ind))
        
    if train_ind < num_train_samples:
        print('Number of train couples in text file: ',num_train_samples)
        print('Number of train couples collected: ',train_ind)
        X_train_chroma = X_train_chroma[:train_ind]
        # X_train_f0img = X_train_f0img[:train_ind]
        X_train_alignedF0 = X_train_alignedF0[:train_ind]
        y_train = y_train[:train_ind]
    
    if test_ind < num_test_samples:
        print('Number of test couples in text file: ',num_test_samples)
        print('Number of test couples collected: ',test_ind)
        X_test_chroma = X_test_chroma[:test_ind]
        # X_test_f0img = X_test_f0img[:test_ind]
        X_test_alignedF0 = X_test_alignedF0[:train_ind]
        y_test = y_test[:test_ind]
    
    with open('train_chroma_X.npy', 'wb') as f:
        np.save(f, X_train_chroma)
    # with open('train_f0img_X.npy', 'wb') as f:
    #     np.save(f, X_train_f0img)
    with open('train_aligned_f0_X.npy', 'wb') as f:
        np.save(f, X_train_alignedF0)    
    with open('train_y.npy', 'wb') as f:
        np.save(f, y_train) 
    
    with open('test_chroma_X.npy', 'wb') as f:
        np.save(f, X_test_chroma)
    # with open('test_f0img_X.npy', 'wb') as f:
    #     np.save(f, X_test_f0img)
    with open('test_aligned_f0_X.npy', 'wb') as f:
        np.save(f, X_test_alignedF0)  
    with open('test_y.npy', 'wb') as f:
        np.save(f, y_test) 
    
    # Storing file names for debugging purposes
    with open('train_test_file_names.pickle', 'wb') as handle:
        pickle.dump((train_comb_files, test_comb_files), handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_dataFrame_f_wr.close()
    train_dataFrame_f_wr.close()

    # Final consistency check for file lists
    check_file_lists(train_comb_files, test_comb_files, train_dataFrame_f, test_dataFrame_f)

    
def check_file_lists(train_comb_files, test_comb_files, train_dataFrame_file, test_dataFrame_file):
    '''Checks file-couples and re-writes train_couples.txt and test_couples.txt 
    since the order is modified'''
    trainDF = pd.read_csv(train_dataFrame_file)
    testDF = pd.read_csv(test_dataFrame_file)
    # check and overwrite train_couples.txt and test_couples.txt since the order is modified
    trainDF['couple']=trainDF['Ref_file']+'.wav '+trainDF['Per_file']+'.wav'
    testDF['couple']=testDF['Ref_file']+'.wav '+testDF['Per_file']+'.wav'
    # Compare lists, re-write list files if all match
    lists_match = True
    for couple1, couple2 in zip(train_comb_files, trainDF['couple']):
        if couple1 != couple2:
            print('Error, file-lists do not match in train split:', couple1, couple2)
            lists_match = False
    if lists_match:
        with open('train_couples.txt','w') as f_out:
            for couple in train_comb_files:
                f_out.write(couple+'\n')

    lists_match = True
    for couple1, couple2 in zip(test_comb_files, testDF['couple']):
        if couple1 != couple2:
            print('Error, file-lists do not match in test split:', couple1, couple2)
            lists_match = False
    if lists_match:
        with open('test_couples.txt','w') as f_out:
            for couple in test_comb_files:
                f_out.write(couple+'\n')    
    

def main():
    # It is assumed that the files melody-data.pickle, train_couples.txt,
    #  test_couples.txt and excluded_references.txt are present at the same 
    #  folder as this script. If not, you should modify the paths below
    script_folder = os.path.abspath(os.getcwd())

    data_file_name = os.path.join(script_folder, 'melody-data.pickle')
    
    train_couples_file = os.path.join(script_folder,'train_couples.txt')
    test_couples_file = os.path.join(script_folder,'test_couples.txt')
    exclude_ref_list_file = os.path.join(script_folder,'excluded_references.txt')
    
    prepare_ML_data(data_file_name, train_couples_file, test_couples_file, exclude_ref_list_file)

            
if __name__ == "__main__":
    main()    


