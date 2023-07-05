# -*- coding: utf-8 -*-
"""
Compares annotations and produces majority voting annotations:
    Anotations are compared using standard metrics such as Krippendorf-alfa as
    well ass machine learning metrics to obtain comparable values with machine 
    learning experiments
    
    
    To gather annotations
    - Labels/grades from different annotators are read from text files, 
    compiled in pandas data frames and saved in the annotation sdata folder 
    ()
    - Merging them in a single pandas dataframe
    
@author barisbozkurt
"""

import os, shutil, zipfile
import matplotlib.pyplot as plt
from scipy import stats as st
from convert_annotations import produce_file_lists
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import agreement
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

LANG = 'TR'
#------------------------------
# Temporarily used function to edit many text files performing text replacement
def text_replace_all_files_in_folder(text_files_folder, replacement_rules={}):
    for root, dirs, files in os.walk(text_files_folder):
        for file in files:
            file = os.path.join(root, file)
            if file.endswith('.txt'):
                # read all lines
                lines = []
                with open(file) as f:
                    for line in f:
                        for source, target in replacement_rules.items():
                            line = line.replace(source, target)
                        lines.append(line)
                # overwrite file
                with open(file, 'w') as f_write:
                    for line in lines:
                        f_write.write(line)
#------------------------------

def create_new_annotation(df, annot_data_folder, method='majority'):
    ''' Creates new annotation files in the annot_data_folder 
    in the format of the original using: majority voting or full-agreement. 
    '''
    
    # Create directories for new files
    new_annot_dir = os.path.join(annot_data_folder,'melody_labels_{}'.format(method))
    new_annot_subdir =  os.path.join(new_annot_dir,'melody_labels_{}'.format(method))
    if not os.path.exists(new_annot_dir):
        os.mkdir(new_annot_dir)
    if not os.path.exists(new_annot_subdir):
        os.mkdir(new_annot_subdir)    
    new_annot_file = os.path.join(new_annot_subdir, 'report_all_files.txt')
    with open(new_annot_file, 'w') as f_write:
        for index, row in df.iterrows():
            if not np.isnan(row[method+'_score']):
                f_write.write('{}\tGrade:{}\n'.format(row['file'], int(row[method+'_score'])))
                    
    # Pack full-agreement annots created by merging all annots
    shutil.make_archive(new_annot_dir, 'zip', new_annot_dir)
    
def compare_annots(df, annot_cols, comparison_results_folder):
    '''Compares labels from different annotators (all couples) and saves 
    results in comparison_results_folder
    '''
    # Create results folder if not exists
    if not os.path.exists(comparison_results_folder):
        os.mkdir(comparison_results_folder)
    
    # Use grades of student performances ('_per' in file name) for comparisons
    df_per = df[df['file'].str.contains('per')]
    
    # Grade labels used in comparison tables
    labels = np.arange(1,5)
    num_annotators = len(annot_cols)

    # Create DataFrame to store comparison-coeffs results
    comp_df = pd.DataFrame()

    # for all annotator couples perfor comparison
    couples = [(annot_cols[ind1],annot_cols[ind2]) for ind1 in range(num_annotators) for ind2 in range(ind1+1,num_annotators)]
    for annotator1, annotator2 in couples:
        # Output file name
        out_file = annotator1 + '_vs_' + annotator2 
        
        # get couple data and drop rows with nan values
        df_sub = df_per[[annotator1, annotator2]].dropna()
        annots_a = df_sub[annotator1].values
        annots_b = df_sub[annotator2].values

        conf_mat = pd.DataFrame(confusion_matrix(annots_a, annots_b,normalize='true'),
                                columns=labels, index=labels)
        conf_mat.index.name = 'Annots. by ' + annotator1
        conf_mat.columns.name = 'Annots. by ' + annotator2
        if LANG == 'TR':
            
            if annotator1[0] == '1' or annotator1[0] == '3':
                annotator1 = annotator1[1:].replace('Asli','Aslı')+'-İlk'
            elif annotator1[0] == '2' or annotator1[0] == '4':
                annotator1 = annotator1[1:].replace('Asli','Aslı')+'-Son'

            if annotator2[0] == '1' or annotator2[0] == '3':
                annotator2 = annotator2[1:].replace('Asli','Aslı')+'-İlk'
            elif annotator2[0] == '2' or annotator2[0] == '4':
                annotator2 = annotator2[1:].replace('Asli','Aslı')+'-Son'

                
            conf_mat.index.name = annotator1 + ' notları'
            conf_mat.columns.name = annotator2 + ' notları'
        
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1.2)
        sns.heatmap(conf_mat, cmap='Blues', annot_kws={'size': 12}, annot=True)
        title = '{} vs {} on files of {} (normalized on rows)'.format(annotator1, annotator2, annotator1)
        if LANG == 'TR':
            title = '{} - {} notları karşılaştırma (örtüşme oranı)'.format(annotator1, annotator2)
        plt.title(title)
        plt.savefig(os.path.join(comparison_results_folder, out_file+'.png'), dpi=300)
        plt.close()
        
        with open(os.path.join(comparison_results_folder, out_file+'.txt'),'w') as f_out:
            
            # Print classification-type comparison using annots_a as true labels and annots_b as predicted labels
            f_out.write(classification_report(annots_a, annots_b))
            
            # Compute agreement coefficients
            formatted_codes = [[1,i,annots_a[i]] for i in range(len(annots_a))] + [[2,i,annots_b[i]] for i in range(len(annots_b))]  
            
            # Rating computation using https://www.nltk.org/_modules/nltk/metrics/agreement.html 
            ratingtask = agreement.AnnotationTask(data=formatted_codes)
            f_out.write("Krippendorff's Alpha:{}\n".format(ratingtask.alpha()))
            f_out.write("Cohen's Kappa:{}\n".format(ratingtask.kappa()))
            
            # Compute MAE
            MAE = np.sum(np.abs(annots_a - annots_b)) / annots_a.size
            f_out.write('MAE on data {} vs {}: {}\n'.format(annotator1, annotator2, MAE))
            
            # Append comparison metric values to dataframe
            comp_df2add = {'Compared_Annots':out_file,'MAE':MAE,
                           'Accuracy':accuracy_score(annots_a, annots_b),
                           'F1_weighted': f1_score(annots_a, annots_b, average='weighted'),
                           'Krippendorf_alfa':ratingtask.alpha(),
                           'Pearson_corr': np.corrcoef(annots_a, annots_b)[0,1]}
            comp_df = comp_df.append(comp_df2add, ignore_index = True)
            
    # Save comparison coeffs to file
    comp_df.to_csv(os.path.join(comparison_results_folder, 'comparison_results.csv'), float_format="%.2f")
#------------------------------

def compare_merge_annots(annot_data_folder, comparison_results_folder):
    merged_file = 'all_annots_2015_2016_mel.txt' # contains all labels from an annotator in a single txt file
    
    # Gather annotation packages, naming convention: melody_labels_*.zip
    data_packages = list()
    for root, dirs, files in os.walk(annot_data_folder):
        for file in files:
            if ('melody_labels_' in file) and (file.endswith('zip')) and ('majority' not in file) and ('fullAgree' not in file): 
                data_packages.append(file)
    # data_packages = list(set(data_packages)) # TODO: delete this line which was put to remove duplicates due to copies of data in subfolders
    
    # Unpacking data from different annotators and merging them 
    #  First create wav_file->grade dictionaries for each annotator, (dict: annotator-> grade dictionary)
    #  then merge all in a single dataframe
    annot_dicts = dict()
    for data_zip_file in data_packages:
        data_folder = data_zip_file.replace('.zip','')
        annotator_name = data_folder.split('_')[-1]
        data_folder = os.path.join(annot_data_folder, data_folder)
        data_zip_file = os.path.join(annot_data_folder, data_zip_file)
        if not os.path.exists(data_folder):  # create folder if not exists and unzip
            os.mkdir(data_folder)
            zip_ref = zipfile.ZipFile(data_zip_file, 'r')
            zip_ref.extractall(data_folder)
            zip_ref.close()
        # merge individual annotation files in folder
        produce_file_lists(data_folder, use_data_folder_4_output=True)
        merged_file_annot = os.path.join(data_folder, merged_file)
        annotations = dict()
        with open(merged_file_annot) as f:
            for line in f:
                if len(line) > 0:
                    line = line.replace('	Grade:', ' ')
                    parts = line.strip().split()
                    annotations[parts[0]] = int(parts[1])
        if ('fullAgree' not in annotator_name): # exclude annotations obtained by merging other annotations
            annot_dicts[annotator_name] = annotations
    
    # Create a common file list first, files not graded by all annotators will be discarded
    all_files = set()
    for annotations in annot_dicts.values():
        all_files = set.union(all_files, annotations.keys())
    all_files = list(all_files)
    data = {'file': all_files}
        
    for annotator, annotations in annot_dicts.items():
        grades = []
        for wav_file in all_files:
            if wav_file in annotations:
                grades.append(annotations[wav_file])
            else:
                grades.append(np.nan)
        data[annotator] = grades
        
    # Create data frame from all dictionaries
    df = pd.DataFrame.from_dict(data)
    annot_cols = list(annot_dicts.keys())
    num_annotators = len(annot_cols)
    
    print('Comparing labels from {} annotators, saving results in {}'.format(num_annotators, comparison_results_folder))
    # Compare annotations and save comparison results to file in folder ../results_plots/
    compare_annots(df, annot_cols, comparison_results_folder)
    
    # add column for full-agreement flag
    df['fullAgree'] = df.loc[:,annot_cols].std(axis=1)==0 # check std of each row to check if all values in row are same
    df['fullAgree_score'] = df.loc[:,annot_cols].mean(axis=1)
    df['fullAgree_score'][~df['fullAgree']] = np.nan
    
    df['majority_score'] = np.nan
    # add column for majority voting grade
    for index, row in df.iterrows():
        mode_val, count = st.mode(row[annot_cols].values)
        if count[0] >= num_annotators/2:
            df.loc[index, ['majority_score']] = mode_val[0]
        
    print('Saving merged annotation files in {}'.format(annot_data_folder))
    df.to_csv(os.path.join(annot_data_folder,'All_annotations.csv'))
    df_ref = df[df['file'].str.contains('ref')]
    df_ref.to_csv(os.path.join(annot_data_folder,'All_annotations_ref.csv'))
    df_per = df[df['file'].str.contains('per')]
    df_per.to_csv(os.path.join(annot_data_folder,'All_annotations_per.csv'))
    
    # Create new annotation files with majority voting and full-agreement
    create_new_annotation(df, annot_data_folder, method='majority')
    create_new_annotation(df, annot_data_folder, method='fullAgree')
    
    # Create grade histogram plots and save in folder ../results_plots/ 
    if LANG == 'TR':
        tr_columns = {'1Asli':'Aslı-İlk',
                      '2Asli':'Aslı-Son',
                      '3Cihan':'Cihan-İlk',
                      '4Cihan':'Cihan-Son',
                      '5Ozan':'Ozan'}
        df_per.rename(columns = tr_columns, inplace = True)
        annot_cols = []
    sns.histplot(data=df_per[sorted(list(tr_columns.values()))], multiple="dodge")
    plt.title('Grade Dist.per annotator')
    if LANG == 'TR':
        plt.title('Uzmanlara göre not dağılımı')
        plt.ylabel('Adet');plt.xlabel('Verilen not')
    # plt.xlabel('Grade (1:completely off, 4:perfect)')
    plt.savefig(os.path.join(comparison_results_folder, 'grade_distribution_annotators.png'), dpi=300)
    plt.close()
    plt.figure()
    if LANG == 'TR':
        df_per.rename(columns = {'fullAgree_score':'Tam uyuşan notlar'}, inplace = True)
        sns.histplot(data=df_per[['Tam uyuşan notlar']], multiple="dodge", legend=False)
        plt.title('Tam uyuşan notlar dağılımı')
        plt.ylabel('Adet');plt.xlabel('Verilen not')
    else:
        sns.histplot(data=df_per[['majority_score','fullAgree_score']], multiple="dodge")
        plt.title('Grade Dist. for majority voting and full-agreement')
    # plt.xlabel('Grade (1:completely off, 4:perfect)')
    plt.savefig(os.path.join(comparison_results_folder, 'grade_distribution_intersection.png'), dpi=300)
    plt.close()
    
def main():
    annot_data_folder = '../../data/melody/'
    comparison_results_folder = '../annot_compare_results'
    
    compare_merge_annots(annot_data_folder, comparison_results_folder)
    
if __name__ == "__main__":
    main()
    