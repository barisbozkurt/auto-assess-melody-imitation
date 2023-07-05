# -*- coding: utf-8 -*-
"""
Regression Experiments on tablular data (trainData.csv and testData.csv files).

@author: barisbozkurt
"""

import os, sys
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('../../') # add root folder to be able to include constants.py
import constants

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression # linear regression added for debugging purposes
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE

def read_balance_normalize_data(data_folder, train_file_path='trainData.csv', test_file_path='testData.csv'):
    ''' Reads csv files, 
        Converts data to numpy arrays that could be fed to ML models
        balances train data using imblearn.over_sampling.SMOTE,
        returns read and preprocessed data
        '''
    # Read train and test data from csv files
    trainDF = pd.read_csv(os.path.join(data_folder, train_file_path))
    testDF = pd.read_csv(os.path.join(data_folder, test_file_path))
    annots_train_files = trainDF[['Per_file', 'grade']].set_index('Per_file').to_dict()['grade']
    annots_test_files = testDF[['Per_file', 'grade']].set_index('Per_file').to_dict()['grade']
    
    # deleting columns that carry file names
    del trainDF['Ref_file'] 
    del trainDF['Per_file']
    
    del testDF['Ref_file']
    del testDF['Per_file']
    
    # All columns except the last contain features, create matrix from data frame content
    X_train_val = trainDF[constants.feature_names].values
    X_test = testDF[constants.feature_names].values

    # Last column contains the grades, convert 1-4 to 0-3
    y_train_val = trainDF.grade.to_numpy() - 1
    y_test = testDF.grade.to_numpy() - 1
    
    # Balance train data
    over_sampler = SMOTE(k_neighbors=2)
    X_train_val, y_train_val = over_sampler.fit_resample(X_train_val, y_train_val)
    
    # Normalize features
    scaler = StandardScaler().fit(X_train_val)
    norm_x_train_val = scaler.transform(X_train_val)
    norm_x_test = scaler.transform(X_test)
    
    return norm_x_train_val, y_train_val, norm_x_test, y_test, annots_train_files, annots_test_files

def run_regression_tests(annot_data_folder, results_folder):

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
    
    regressors2test = [LinearRegression(), RandomForestRegressor(random_state=0), XGBRegressor(), AdaBoostRegressor()]
    
    regressor_names = ['Linear regression','Random forest', 'XGBoost','Adaboost']
    
    report_file = open(os.path.join(results_folder,"results_regression.txt"), "w")
    
    print('Running regression tests for data folders:', data_folders)
    results_df = pd.DataFrame() # data frame to store overall results
    for data_folder in data_folders:
      report_file.write('--------------------\n')
      report_file.write('RUNNING TESTS FOR DATA IN {}\n'.format(data_folder))
      norm_x_train_val, y_train_val, norm_x_test, y_test, annots_train_files, annots_test_files = read_balance_normalize_data(data_folder)
    
      for name, regressor2test in zip(regressor_names, regressors2test):
    
        # Running single experiment: training with train_val and testing with test data
        report_file.write('--------------{}--------------\n'.format(name))
        regressor2test.fit(norm_x_train_val, y_train_val)
        y_pred = regressor2test.predict(norm_x_test)
        
        MAE = np.sum(np.abs(y_pred-y_test))/y_test.size
        y_pred_round = np.round(y_pred)
        MAE_rounded = np.sum(np.abs(y_pred_round-y_test))/y_test.size
        acc = accuracy_score(y_test, y_pred_round)
        f1_weighted = f1_score(y_test, y_pred_round, average='weighted')
        
        report_file.write('MAE:{}\n'.format(MAE))
        report_file.write('MAE with rounded grades:{}\n'.format(MAE_rounded))
        
        # Append comparison metric values to dataframe
        df2add = {'Data':data_folder, 'Model':name,'MAE':MAE,'MAE_rounded':MAE_rounded, 'Accuracy':acc,'F1_weighted': f1_weighted}
        results_df = results_df.append(df2add, ignore_index = True)
        
    report_file.close()

    # Save overallresults to file
    results_df.to_csv(os.path.join(results_folder, 'regression_results.csv'), float_format="%.2f")
    
    print('Regression test results saved to folder:', results_folder)

def main():
    annot_data_folder = '../../../data/annotations/'
    print('Running regression tests for data in', annot_data_folder)
    results_folder = 'results'
    run_regression_tests(annot_data_folder, results_folder)
    print('Test results written to "results" folder')
    
if __name__ == "__main__":
    main()