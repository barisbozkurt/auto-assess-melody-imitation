# -*- coding: utf-8 -*-
"""
Classifier Experiments on tablular data (trainData.csv and testData.csv files).
@author barisbozkurt
"""

import os, sys
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # adding logistic regression for debugging purposes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.svm import SVC

sys.path.append('../../') # add root folder to be able to include constants.py
import constants


feature_names = constants.feature_names

def read_prep_csv(data_folder, train_file_path='trainData.csv', test_file_path='testData.csv'):
  
    # Read train and test data from csv files
    trainDF = pd.read_csv(os.path.join(data_folder, train_file_path))
    testDF = pd.read_csv(os.path.join(data_folder, test_file_path))
    # delete unnamed columns (these are extra columns containing no information)
    for col_name in trainDF.columns:
        if 'Unnamed' in col_name:
            del trainDF[col_name]
    for col_name in testDF.columns:
        if 'Unnamed' in col_name:
            del testDF[col_name]
    
    # annots_train_files = trainDF[['Per_file', 'grade']].set_index('Per_file').to_dict()['grade']
    # annots_test_files = testDF[['Per_file', 'grade']].set_index('Per_file').to_dict()['grade']
    
    trainDF['Question'] = pd.Categorical(trainDF['Ref_file'].apply(lambda x:'_'.join(x.split('_')[:2])))
    trainDF['QuestionID'] = trainDF['Question'].cat.codes
    
    testDF['Question'] = pd.Categorical(testDF['Ref_file'].apply(lambda x:'_'.join(x.split('_')[:2])))
    testDF['QuestionID'] = testDF['Question'].cat.codes
    # deleting columns that carry file names
    for col_2_del in ['Ref_file', 'Per_file', 'Question']:
        del trainDF[col_2_del]
        del testDF[col_2_del]
    return trainDF, testDF 

def preproc_train(trainDF):
    X_train = trainDF[feature_names].values
    y_train = trainDF.grade.to_numpy() - 1
    
    # Balance train data
    over_sampler = SMOTE(k_neighbors=2)
    X_train, y_train = over_sampler.fit_resample(X_train, y_train)
    
    # Normalize features: this step is not needed for decision trees
    scaler = StandardScaler().fit(X_train)
    norm_x_train = scaler.transform(X_train)
    return norm_x_train, y_train, scaler

def cross_validation_with_group_split(model_name, trainDF, groupColumn, target_names, f_out, num_splits=5):
    '''Given the train data as a pandas data frame and the model,
    create splits using the QuestionID column and runs leave-one-group-out cross-validation'''
    
    # create a new train and test DF from trainDF
    cross_valid_accs = []
    
    groupIDs = trainDF[groupColumn].unique()
    size_split = groupIDs.size // num_splits
    group_set_inds = [(i*size_split, (i+1)*size_split) for i in range(num_splits)] # + [(num_splits*size_split, groupIDs.size)]
    # print(group_set_inds)
    for group_set_ind in group_set_inds:
        group_set = groupIDs[group_set_ind[0]:group_set_ind[1]]
        group = trainDF[groupColumn].apply(lambda x: x in group_set) 
        train_splitDF = trainDF[~group]
        val_splitDF = trainDF[group]
        classifier2test = RandomForestClassifier(random_state=0)
        acc, f1_weighted, MAE = run_test(model_name, '','', classifier2test, train_splitDF, val_splitDF, target_names, f_out, fullReport=False)
        cross_valid_accs.append(acc)
        f_out.write('Validation split question ids:{} train shape:{} valid shape:{} val-accuracy:{}\n'.format(group_set, train_splitDF.shape, val_splitDF.shape, acc))
 
    
    return np.array(cross_valid_accs)

def run_test(model_name, data_folder, results_folder, classifier2test, trainDF, testDF, target_names, f_out, fullReport=True):
    '''Trains model with complete train data and reports results on test data'''
    norm_x_train, y_train, scaler = preproc_train(trainDF)
    classifier2test.fit(norm_x_train, y_train)
    
    X_test = testDF[feature_names].values
    y_test = testDF.grade.to_numpy() - 1
    norm_x_test = scaler.transform(X_test)
    y_pred = classifier2test.predict(norm_x_test)
    labels = np.arange(1,5)
    
    # Reporting evaluation results
    if fullReport:
        conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize='true'), columns=labels, index=labels)
        conf_mat.index.name = 'True value'
        conf_mat.columns.name = 'Predicted value'
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1.2)
        sns.heatmap(conf_mat, cmap='Blues', annot_kws={'size': 12}, annot=True)
        annotator = data_folder.split('_')[-1]
        plt.title('Data: ' + annotator + ', model: ' + model_name)
        plt.savefig(os.path.join(results_folder, model_name+'_'+annotator+'_confusionMat.png'), dpi=300)
        report = classification_report(y_test, y_pred, target_names=target_names)
        f_out.write(report+'\n')

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    MAE = np.sum(np.abs(y_test - y_pred)) / y_pred.size
    return acc, f1_weighted, MAE

def run_classification_tests(annot_data_folder, results_folder):

    # Unpacking data from different annotators
    data_packages = list()
    for root, dirs, files in os.walk(annot_data_folder):
        for file in files:
            if ('melody_data4ML_' in file) and (file.endswith('zip')) and ('fullAgree' not in file): # fullAgree set does not have enough samples for grades 2 and 3  
                data_packages.append(file)
    
    data_folders = list()
    for data_zip_file in data_packages:
        data_folder = os.path.join(annot_data_folder, data_zip_file.replace('.zip',''))
        data_folders.append(data_folder)
        data_zip_file = os.path.join(annot_data_folder, data_zip_file)
        if not os.path.exists(data_folder):  # create folder if not exists and unzip
            os.mkdir(data_folder)
            zip_ref = zipfile.ZipFile(data_zip_file, 'r')
            zip_ref.extractall(data_folder)
            zip_ref.close()
    
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
        
    classifiers2test = [LogisticRegression(), RandomForestClassifier(random_state=0), XGBClassifier(), SVC(kernel='rbf', gamma='auto')]
    classifier_names = ['LogisticRegression','RandomForest','XGBoost','SVM']
    
    labels = np.arange(1,5)
    labels_str = [str(val) for val in labels]
    
    groupColumn = 'QuestionID'
    
    print('Running classification tests for data folders:', data_folders)
    num_cross_val_tests = 5
    results_df = pd.DataFrame() # data frame to store overall results
    with open(os.path.join(results_folder, 'classification_results.txt'), 'w') as f_out:
        for data_folder in data_folders:
            # data_folder = os.path.join(annot_data_folder, data_folder)
            f_out.write('--------------------\n')
            f_out.write('RUNNING TESTS FOR {}\n'.format(data_folder))
            trainDF, testDF = read_prep_csv(data_folder)
            scores = {}
            f_out.write('Cross-validation (on splits with respect to melody group)\n')
            for model_name, classifier2test in zip(classifier_names, classifiers2test):
                f_out.write('Running test for {}\n'.format(model_name))
                scores[model_name] = cross_validation_with_group_split(model_name, trainDF, groupColumn, labels_str, f_out, num_splits=num_cross_val_tests)
                f_out.write('{}\t, scores:{}\tmean:{}\n'.format(model_name, scores[model_name], np.mean(scores[model_name])))
              
                # Running single experiment: training with complete train data and testing 
                #  with test data
                f_out.write('--------------{} training an all train-data and testing on test data --------------\n'.format(model_name))
                acc, f1_weighted, MAE = run_test(model_name, data_folder, results_folder, classifier2test, trainDF, testDF, labels_str, f_out)
                
                # Append comparison metric values to dataframe
                df2add = {'Data':data_folder.split('_')[-1], 'Model':model_name,'MAE':MAE, 'Accuracy':acc,'F1_weighted': f1_weighted}
                results_df = results_df.append(df2add, ignore_index = True)
    
    # Save overallresults to file
    results_df.to_csv(os.path.join(results_folder, 'classification_results.csv'), float_format="%.2f")
    
    print('Classifier test results saved to folder:', results_folder)
        
def main():
    annot_data_folder = '../../../data/annotations/'
    print('Running classification tests for data in', annot_data_folder)
    results_folder = 'results'
    run_classification_tests(annot_data_folder, results_folder)
    print('Test results written to "results" folder')
    
if __name__ == "__main__":
    main()