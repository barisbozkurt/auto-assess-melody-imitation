#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs all processes below:
    - Download audio data from Zenodo
        - Convert .m4a files to .wav
        - Move all feature files in the same folder as audio files
    - Compare annotations, produce reports in data/annotations_comparison_results
    - Run data preprocessing scripts to extract features and create tabular data for machine learning experiments
    - Run machine learning experiments
        - Classifier tests
        - Regressor tests

@author: barisbozkurt
"""
import os, shutil, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/dataProcessing")
import zipfile
import time
from convert_annotations import produce_file_lists, create_test_train_split_file_lists
from compare_merge_annotations import compare_merge_annots
from datapreprocess_melody import create_recording_objects
from group_melody_features import prepare_ML_data
from resample_f0series import resample_f0files_inFolder
from downloadAudioFromZenodo import download_melody_data_from_zenodo

def consistency_check_annots(tabular_data_files, annotations_file):
    '''
    Checks consistency of tabular data and the annotation files
    Used for debugging purposes.

    Parameters
    ----------
    tabular_data_files : list of str
        Path to tabular data.
    annotations_file : str
        Path to the single annotation file that contains all annotations.

    Returns
    -------
    None.

    '''
    num_errors = 0
    # Read annoations file into a dict mapping filename-> grade
    annots_dict = {}
    with open(annotations_file) as list_file:
        for line in list_file:
            file = line.split()[0].strip().replace('.wav','')
            grade = int(line.strip().split('Grade:')[-1])
            annots_dict[file] = grade
    
    # Read ML data, check if annotations match
    # Also check if question matches for ref-per files
    for tabular_data_file in tabular_data_files:
        line_ind = 0
        with open(tabular_data_file) as list_file:
            for line in list_file:
                if line_ind != 0: # skip first line
                    line_splits = line.split(',')
                    ref_file = line_splits[0].strip()
                    per_file = line_splits[1].strip()
                    grade = int(line_splits[-1].strip())
                    
                    # Check consistency of question/pattern
                    pattern_ref = '_'.join(ref_file.split('_')[:2])
                    pattern_per = '_'.join(per_file.split('_')[:2])
                    if pattern_ref != pattern_per:
                        num_errors += 1
                        print('!Error, question/pattern do not match')
                        print('Line {} in {}'.format(line_ind, tabular_data_file))
                    
                    # Checking consistency of grades in two files
                    if grade != annots_dict[per_file]:
                        num_errors += 1
                        print('!Error, grades do not match')
                        print('Line {} in {} has grade {}'.format(line_ind, tabular_data_file, grade))
                        print(' where {} for file {} has grade {}'.format(annotations_file, per_file, annots_dict[per_file]))
                
                line_ind += 1
    if num_errors == 0:
        print('Consistency of annotations and final tabular data checked: OK')
    
def exclusivity_check_ML_data(test_data_file, train_data_file, check_ref_files=False):
    '''
    Checks exclusivity of test and train sets
    Used for debugging purposes.
    
    Parameters
    ----------
    test_data_file : str
        Path to test data.
    train_data_file : str
        Path to train data.
    check_ref_files : bool, optional
        Flag to check the reference files exclusivity. The default is False.

    Returns
    -------
    None.

    '''
    num_errors = 0
    test_ref_files = []
    test_per_files = []
    line_ind = 0
    with open(test_data_file) as list_file:
        for line in list_file:
            if line_ind != 0: # skip first line
                line_splits = line.split(',')
                test_ref_files.append(line_splits[0].strip())
                test_per_files.append(line_splits[1].strip())
            line_ind += 1
    
    line_ind = 0
    with open(train_data_file) as list_file:
        for line in list_file:
            if line_ind != 0: # skip first line
                line_splits = line.split(',')
                train_ref_file = line_splits[0].strip()
                train_per_file = line_splits[1].strip()
                # Checking exclusivity of student performance files
                if train_per_file in test_per_files:
                    num_errors += 1
                    print('Error:')
                    print(' Train per file exists also in test set:', train_per_file)
                # Checking exclusivity of teacher/reference files
                if check_ref_files:
                    if train_ref_file in test_ref_files:
                        num_errors += 1
                        print('Error:')
                        print(' Train ref file exists also in test set:', train_ref_file)

            line_ind += 1
            
    if num_errors == 0:
        print('Exclusivity of train and test data checked: OK')

                
#---------------------------
def main():
    start_time = time.time()
    
    root_data_folder = '../data/'
    annot_data_folder = root_data_folder + 'annotations/'
    audio_data_folder = root_data_folder + 'audio_feature_files/'
    
    # Running data download from Zenodo
    
    # Mode selection as below short-circuits audio format conversion, 
    #   To get real .wav files, set it to 'convertM4aToWav'
    mode ='renameM4aToWav' 
    download_melody_data_from_zenodo(root_data_folder, mode=mode)
    
    # Compare labels from different annotators and merge them to create
    #   majority voting and full-agreement annotations
    comparison_results_folder = '../data/annotations_comparison_results'
    compare_merge_annots(annot_data_folder, comparison_results_folder)
    
    # Gather annotation packages, naming convention: melody_labels_*.zip
    annotation_file_packages = list()
    for root, dirs, files in os.walk(annot_data_folder):
        for file in files:
            if ('melody_labels_' in file) and (file.endswith('zip')): 
                annotation_file_packages.append(file)
    
    # List of files to be created and finally packed in the output package
    files_to_pack_in_ML_data = ['all_annots_2015_2016_mel.txt', # incl. all grades per file
                                'excluded_references.txt', # incl. list of excluded references (grade < 4)
                                'testData.csv', # incl. tabular data containing distance features and grade
                                'trainData.csv', # incl. tabular data containing distance features and grade
                                'train_chroma_X.npy', # incl. chroma-pairs (train-set)
                                #'train_f0img_X.npy', # incl. f0image-pairs (train-set) # this feature is disabled, was not useful in tests
                                'train_y.npy', # incl. outputs/targets/grades (train-set)
                                'test_chroma_X.npy', # incl. chroma-pairs (test-set)
                                #'test_f0img_X.npy', # incl. f0image-pairs (test-set) # this feature is disabled, was not useful in tests
                                'test_y.npy', # incl. outputs/targets/grades (test-set)
                                'train_test_file_names.pickle',
                                'melody-data.pickle', # incl. all Recording objects
                                'test_couples.txt',
                                'train_couples.txt',
                                'test_couples_per_grade.pickle',
                                'train_couples_per_grade.pickle',
                                'train_aligned_f0_X.npy', # incl. aligned f0-cent pairs (train-set)
                                'test_aligned_f0_X.npy'] # incl. aligned f0-cent pairs (test-set)
    
    script_folder = os.path.abspath(os.getcwd())
    
    # Creating feature files and tabular data for each annotator
    for annot_ind, annotation_file_package in enumerate(annotation_file_packages):
        annotator_name = annotation_file_package.split('.')[0].split('_')[-1]
        print('-'*40)
        print('PREPARING DATA OF ANNOTATOR ', annotator_name, ':', annotation_file_package)
        
        # Unzip annotation files package
        unzipped_annot_folder = annot_data_folder + 'temp/'
        with zipfile.ZipFile(os.path.join(annot_data_folder, annotation_file_package),
                              'r') as zip_ref:
            zip_ref.extractall(unzipped_annot_folder)
    
        # Merge annotation files and create test-train splits    
        produce_file_lists(unzipped_annot_folder)
        create_test_train_split_file_lists(script_folder)
        
        # If previous splits exist, overwrite train_couples.txt and test_couples.txt files
        splits_pack_file = os.path.join(annot_data_folder, 'splits_'+annotator_name+'.zip')
        if os.path.exists(splits_pack_file):
            with zipfile.ZipFile(splits_pack_file, 'r') as zip_ref:
                zip_ref.extractall(unzipped_annot_folder)
            for file_name in ['test_couples.txt', 'train_couples.txt']:
                shutil.copyfile(os.path.join(unzipped_annot_folder, file_name), 
                                os.path.join(script_folder, file_name))
        
        # Delete the temporary folder created for annotation files
        shutil.rmtree(unzipped_annot_folder)
        
        # Creating recording objects: runs f0 estimation and chroma computation
        #  if *.f0.npy and *.chroma.npy files do not exist
        # Creates the following files in the same folder as wav files:
        #   *.f0.npy: f0 series in Hz
        #   *.f0_cent.npy: f0 series in cents (ref. freq. defined in constants.py)
        #   *.chroma.npy: chromagram with dimensions defined in constants.py
        #   *.f0img.npy: image representation for f0-cents with dimensions defined in constants.py # this feature is disabled, was not useful in tests
        target_rec_objects_file = os.path.join(script_folder, 'melody-data.pickle')
        annotations_file = 'all_annots_2015_2016_mel.txt'
        create_recording_objects(target_rec_objects_file, audio_data_folder, annotations_file)
        
        # Create resampled versions of f0-series data
        if annot_ind == 0: # no need to re-run for each annotation 
            resample_f0files_inFolder(audio_data_folder)
        
        # Grouping data and preparing data files ready for ML experiments
        # Creates the following files:
            
        train_couples_file = os.path.join(script_folder,'train_couples.txt')
        test_couples_file = os.path.join(script_folder,'test_couples.txt')
        exclude_ref_list_file = os.path.join(script_folder,'excluded_references.txt')
        prepare_ML_data(target_rec_objects_file, train_couples_file, test_couples_file, exclude_ref_list_file)
        
        # CONSISTENCY CHECKS IN CREATE FILES
        # Checks consistency of annotations and final tabular data
        consistency_check_annots(['testData.csv','trainData.csv'], annotations_file)
        # Check exclusivity of train and test data
        exclusivity_check_ML_data('testData.csv','trainData.csv', check_ref_files=True)
        
        # Packing ML data files in a zip package
        ML_data_folder = annot_data_folder + 'melody_data4ML/'
        os.mkdir(ML_data_folder)
        for file in files_to_pack_in_ML_data:
            shutil.copyfile(file, os.path.join(ML_data_folder, file))
        
        zip_file_name = 'melody_data4ML_'+ annotator_name
    
        shutil.make_archive(os.path.join(annot_data_folder, zip_file_name), 'zip', ML_data_folder)
        
        # Delete ML data folder
        shutil.rmtree(ML_data_folder)
        
        # Delete all csv npy pickle and text files in scripts folder 
        #  (these are intermediate files)
        for ext in ['.csv','.npy','.pickle','.txt']:
            os.system('rm {}/*{}'.format(script_folder, ext))        
    
    stop_time = time.time()
    print('Total duration for data preparation: ', (stop_time - start_time)/60, 'minutes')
    
    # Running classifier tests, results will be put in "results" folder
    os.chdir('ML_experiments/classifier_tests/')
    os.system('python classificationtests_melody.py')

    # Running regression tests, results will be put in "results" folder
    os.chdir('../regression_tests/')
    os.system('python regressiontests_melody.py')
if __name__ == "__main__":
    main()