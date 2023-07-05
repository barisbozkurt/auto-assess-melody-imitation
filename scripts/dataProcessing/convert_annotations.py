#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads melody annotation files, converts and saves them in new file-list files

Performs split for test and train and produces list files for those

@author: barisbozkurt
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import random

def rename_files(dba_folder, search_str, replace_str):
    '''Renames all files in a given directory
    Used for renaming manually corrected onset files: .txt -> .os.txt
    '''
    for root, dirs, files in os.walk(dba_folder):
        for file in files:
            if search_str in file: # annotation files starts with 'report'
                file_path = os.path.join(root, file)
                target_path = file_path.replace(search_str, replace_str)
                os.rename(file_path, target_path)
                
def produce_file_lists(dba_folder, use_data_folder_4_output=False):
    '''
    Given the annotation files produced by the annotation software in subfolders
    of dba_folder, produces list files

    Parameters
    ----------
    dba_folder : str
        Path to main folder containing annotation files.

    Returns
    -------
    None
    Produces new text files containing list of files

    '''
    out_file = 'all_annots_2015_2016_mel.txt'
    exc_file = 'excluded_references.txt'
    list_file_ref = 'listreferences.txt'
    list_file_per = 'listperformances.txt'
    grd_4_per_files = 'listperformances_grd4.txt'
    grd_3_per_files = 'listperformances_grd3.txt'
    grd_2_per_files = 'listperformances_grd2.txt'
    grd_1_per_files = 'listperformances_grd1.txt'
    
    # added option to select output directory as the data source directory
    if use_data_folder_4_output: 
        out_file = os.path.join(dba_folder, out_file)
        exc_file = os.path.join(dba_folder, exc_file)
        list_file_ref = os.path.join(dba_folder, list_file_ref)
        list_file_per = os.path.join(dba_folder, list_file_per)
        grd_4_per_files = os.path.join(dba_folder, grd_4_per_files)
        grd_3_per_files = os.path.join(dba_folder, grd_3_per_files)
        grd_2_per_files = os.path.join(dba_folder, grd_2_per_files)
        grd_1_per_files = os.path.join(dba_folder, grd_1_per_files)
    
    out_file_wr = open(out_file, 'w')
    exc_file_wr = open(exc_file, 'w')
    list_file_ref_wr = open(list_file_ref, 'w')
    list_file_per_wr = open(list_file_per, 'w')
    grd_4_per_files_wr = open(grd_4_per_files, 'w')
    grd_3_per_files_wr = open(grd_3_per_files, 'w')
    grd_2_per_files_wr = open(grd_2_per_files, 'w')
    grd_1_per_files_wr = open(grd_1_per_files, 'w')
    
    grd_4_per_files_list = []
    grd_3_per_files_list = []
    grd_2_per_files_list = []
    grd_1_per_files_list = []
    
    num_performances = 0
    num_perf_per_grade = [0,0,0,0] # number of performances per grade
    
    annot_dict = {}
    for root, dirs, files in os.walk(dba_folder):
        for file in files:
            if 'report' in file: # annotation files starts with 'report'
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    for line in f:
                      audio_file = line.strip().split()[0].split('/')[-1]
                      audio_file = audio_file.split("\\")[-1] # removing folder info from filename(windows)
                      grade = int(line.strip().split()[-1].split('Grade:')[-1])
                      annot_dict[audio_file] = grade
                      if 'ref' in audio_file and grade < 4:
                          exc_file_wr.write(audio_file + '\tGrade:' + str(grade) + '\n')                          
                      else:
                          out_file_wr.write(audio_file + '\tGrade:' + str(grade) + '\n')
                          if 'per' in audio_file:
                              num_performances += 1
                              num_perf_per_grade[grade-1] += 1 
                              list_file_per_wr.write(audio_file + '\n')
                              if grade == 4:
                                  grd_4_per_files_wr.write(audio_file + '\n')
                                  grd_4_per_files_list.append((audio_file, grade))
                              elif grade == 3:
                                  grd_3_per_files_wr.write(audio_file + '\n')
                                  grd_3_per_files_list.append((audio_file, grade))
                              elif grade == 2:
                                  grd_2_per_files_wr.write(audio_file + '\n')
                                  grd_2_per_files_list.append((audio_file, grade))
                              elif grade == 1:
                                  grd_1_per_files_wr.write(audio_file + '\n')
                                  grd_1_per_files_list.append((audio_file, grade))
                          if 'ref' in audio_file:
                              list_file_ref_wr.write(audio_file + '\n')
                  
    # print('Number of student performances:', num_performances)
    out_file_wr.close()
    list_file_ref_wr.close()
    list_file_per_wr.close()
    exc_file_wr.close()
    
    grd_4_per_files_wr.close()
    grd_3_per_files_wr.close()
    grd_2_per_files_wr.close()
    grd_1_per_files_wr.close()
    
    # Report counts per grade
    # for grade in range(1,5):
    #     print('Number of grade {} performances: {}'.format(grade, num_perf_per_grade[grade-1]))
    

def write_list_to_file(list_2_write, file_path):
    f_wr = open(file_path, 'w')
    for element in list_2_write:
        f_wr.write(element + '\n')
    f_wr.close()
    

    
def create_test_train_split_file_lists(list_files_folder, test_split_ratio=0.25 ):
    '''
    Create test-train split on questions/patterns level.
    
    Steps:
        - Find minimum number of grade occurence and decide num test applying the
        test_split_ratio to this number
        - Find least_freq_grade and get count per patterns for this grade
        - Pick patterns with high number of examples for the least_freq_grade as
        test patterns (this guarantees picking least number of patterns that 
        provide enough samples for the least_freq_grade)
        - Create test couples for these patterns and balance data wrt grade
        - Create train couples
    
    '''
    # Create output folder if not exists
    if not os.path.exists(list_files_folder):
        os.mkdir(list_files_folder)
        
    num_files_per_grade = {}
    file_list_per_grade = {}
    for grade in [1,2,3,4]:
        list_file = os.path.join(list_files_folder, 'listperformances_grd{}.txt'.format(grade))
        file_list = []
        with open(list_file) as file_reader:
            for line in file_reader:
                if len(line) > 2:
                    file_list.append(line.strip())
        num_files_per_grade[grade] = len(file_list)
        file_list_per_grade[grade] = file_list
        
    #Find minimum number of grade occurence
    min_num_files = min(num_files_per_grade.values())
    # Find files of the least freq. grade
    least_freq_grade = [grade for grade, num_files in num_files_per_grade.items() if num_files == min_num_files][0]
    pattern_names = ['_'.join(file_name.split('_')[:2]) for file_name in file_list_per_grade[least_freq_grade]]
    # count num files per pattern for the leat freq grade
    pattern_counter = {} # pattern->count
    for pattern in pattern_names:
        if pattern in pattern_counter:
            pattern_counter[pattern] += 1
        else:
            pattern_counter[pattern] = 1
    
    #Define number of files to keep for test set for each grade
    # the rest will be kept for train (later balanced before ML tests)
    num_test_files_per_grade = int(min_num_files * test_split_ratio)
    
    # Select patterns with highest number of files to decide the patterns for 
    #  the test split 
    sorted_counts = sorted(pattern_counter.values())[::-1]
    total = 0
    selected_counts = []
    selected_patterns = []
    for count in sorted_counts:
        if total < num_test_files_per_grade:
            selected_counts.append(count)
            total += count
            sel_pattern = [pattern for pattern, num in pattern_counter.items() if num == count][0]
            pattern_counter[sel_pattern] = 0 # reset to avoid duble selection
            selected_patterns.append(sel_pattern)
    print('Patterns selected for the test set:', selected_patterns)
    
    #Pick test files (performances)
    test_files_per_grade = {}
    test_files = []
    for grade, file_list in file_list_per_grade.items():
        selected_list = []
        for file_name in file_list:
            pattern = '_'.join(file_name.split('_')[:2])
            if pattern in selected_patterns:
                selected_list.append(file_name)
        
        test_files_per_grade[grade] = selected_list
        test_files += selected_list
    
    #Pick train files (performances)
    train_files_per_grade = {}
    train_files = []
    for grade, file_list in file_list_per_grade.items():
        selected_list = []
        for file_name in file_list:
            pattern = '_'.join(file_name.split('_')[:2])
            if pattern not in selected_patterns:
                selected_list.append(file_name)
        
        train_files_per_grade[grade] = selected_list
        train_files += selected_list   
    
    # Check if any test file appears in train file list
    for file in test_files:
        if file in train_files:
            print('Error in splitting, file exists in both train and test:', file)
            
    # Choose reference files to couple with performance files

    #Read references file list
    ref_file_list = []
    with open(os.path.join(list_files_folder,'listreferences.txt')) as file_reader:
        for line in file_reader:
            if len(line) > 2:
                ref_file_list.append(line.strip())
    
    # Pick test ref files and form test ref-per couples
    test_ref_files = []
    test_couples_per_grade = {1:[], 2:[], 3:[], 4:[]}
    num_couples_per_grade = [0,0,0,0]
    for grade, file_list in test_files_per_grade.items():
        for file in file_list:
            question_id = '_'.join(file.split('_')[:2])
            for ref_file in ref_file_list:
                question_id_ref = '_'.join(ref_file.split('_')[:2])
                if question_id == question_id_ref:
                    test_couples_per_grade[grade].append(ref_file+'\t'+file)
                    num_couples_per_grade[grade-1] += 1
                    if ref_file not in test_ref_files:
                        test_ref_files.append(ref_file)
    # select a balanced set for test couples:
    min_num_couples = min(num_couples_per_grade)
    for grade, file_list in test_couples_per_grade.items():
        if len(file_list) > min_num_couples:
            random.shuffle(file_list)
            test_couples_per_grade[grade] = file_list[:min_num_couples]
            
            
    # Train ref files will be all except test ref files
    train_ref_files = []
    for ref_file in ref_file_list:
        if ref_file not in test_ref_files:
            train_ref_files.append(ref_file)
            
    # Create ref-per couples for train
    train_couples_per_grade = {1:[], 2:[], 3:[], 4:[]}
    for grade, file_list in train_files_per_grade.items():
        for file in file_list:
            question_id = '_'.join(file.split('_')[:2])
            for ref_file in train_ref_files:
                question_id_ref = '_'.join(ref_file.split('_')[:2])
                if question_id == question_id_ref:
                    train_couples_per_grade[grade].append(ref_file+'\t'+file)
    
    # Check if any test-ref file appears in train-ref file list
    for file in test_ref_files:
        if file in train_ref_files:
            print('Error in splitting, file exists in both train and test:', file)
    
    # Writing lists to text files
    write_list_to_file(test_files, 'test_per_files.txt')
    write_list_to_file(train_files, 'train_per_files.txt')
    write_list_to_file(test_ref_files, 'test_ref_files.txt')
    write_list_to_file(train_ref_files, 'train_ref_files.txt')    
    # Collapse dictionaries to list to write to file
    test_couples_list = []
    for grade, couple_list in test_couples_per_grade.items():
        test_couples_list += couple_list
    train_couples_list = []
    for grade, couple_list in train_couples_per_grade.items():
        train_couples_list += couple_list
    
    write_list_to_file(test_couples_list, 'test_couples.txt')
    write_list_to_file(train_couples_list, 'train_couples.txt')
    
    with open('test_couples_per_grade.pickle', 'wb') as handle:
        pickle.dump(test_couples_per_grade, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('train_couples_per_grade.pickle', 'wb') as handle:
        pickle.dump(train_couples_per_grade, handle, protocol=pickle.HIGHEST_PROTOCOL)    

def main():
    # Before running, first unzip /data/melody/melody_labels_Asli.zip 
    # to a folder which should produce the folder below
    dba_folder = '../../data/melody/melody_labels_Cihan/'
    # Produce file lists
    produce_file_lists(dba_folder)
    
    # Perform test-train split and produce new files
    # New files are created at the same location as this script to avoid 
    # overwriting files in the data folder and providing chance to check them 
    create_test_train_split_file_lists(os.path.abspath(os.getcwd()))
    
if __name__ == "__main__":
    main()