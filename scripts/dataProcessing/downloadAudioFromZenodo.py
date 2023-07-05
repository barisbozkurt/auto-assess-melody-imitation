#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download audio data from 
https://zenodo.org/record/8007358
and convert to .wav

In addition to audio data, the file in Zenodo includes feature files. The code
for feature extraction is available in this folder. Yet, feature extraction is
time-consuming. Hence, the pre-computed feature files are unzipped and copied 
to audio folder. The feature extraction steps skip extraction if a feature file
already exists. To re-compute features, simply delete all feature files and run
feature_extractor.py

@author: barisbozkurt
"""
import os
import shutil
import urllib.request
from pydub import AudioSegment

def move_all_files(source_folder, destination_folder, file_extention='', 
                   delete_source_folder=True, mode='move',
                   text_replace_in_file={}):
    '''Moves all files in one directory to another '''
    # gather all files
    allfiles = os.listdir(source_folder)
     
    # iterate on all files to move them to destination folder
    for f in allfiles:
        if file_extention in f:
            src_path = os.path.join(source_folder, f)
            if text_replace_in_file:
                for source, target in text_replace_in_file.items():
                    f = f.replace(source, target)
            dst_path = os.path.join(destination_folder, f)
            if mode=='move':
                shutil.move(src_path, dst_path)
            elif mode=='copy':
                shutil.copy(src_path, dst_path)
    
    # delete source folder once emptied
    if delete_source_folder:
        os.removedirs(source_folder)


def download_melody_data_from_zenodo(target_folder, mode='convertM4aToWav'):
    '''
    Downloads rhythm data from Zenodo. Use it if you don't have the rhythm data

    Returns
    -------
    None. Creates a "data" folder and puts all audio there

    '''
    #Download data from Zenodo : https://zenodo.org/record/8007358
    file_url = "https://zenodo.org/record/8007358/files/MASTmelody_dataset.zip?download=1"
    zip_file_name = file_url.split('/')[-1].split('?')[0]
    
    if not os.path.exists(zip_file_name):
        print('Downloading 331.2Mb zip file fom Zenodo to folder of this script')
        urllib.request.urlretrieve(file_url, zip_file_name)
    else:
        print("Data file exists, unzipping ...")
    
    #Unpacking the data zip package
    zip_file_name = file_url.split('/')[-1].split('?')[0]
    shutil.unpack_archive(zip_file_name, target_folder)
    # os.remove(zip_file_name)
    
    # Zip file contains a subfolder audioFiles 
    audio_feature_files_folder = target_folder + 'audio_feature_files/'
    
    # Audio files will be converted to wav and placed in audio_feature_files_folder
    #  as well as the feature files
    if not os.path.exists(audio_feature_files_folder):
        os.mkdir(audio_feature_files_folder)    
    
    # Unpack audio files in audioFiles/MAST_melody_audio.zip
    audio_file_pack = os.path.join(target_folder, "audioFiles/MAST_melody_audio.zip")
    shutil.unpack_archive(audio_file_pack, audio_feature_files_folder)
    source_folder = audio_feature_files_folder+"MAST_melody_audio/" 
    move_all_files(source_folder, audio_feature_files_folder)
    
    
    # Unpack chroma files in chroma/MAST_melody_chroma.zip
    chroma_file_pack = os.path.join(target_folder, "chroma/MAST_melody_chroma.zip")
    shutil.unpack_archive(chroma_file_pack, audio_feature_files_folder)
    source_folder = audio_feature_files_folder+"MAST_melody_chroma/" 
    move_all_files(source_folder, audio_feature_files_folder)
    
    # Unpack f0 files in f0data_crepe/MAST_melody_f0.zip
    f0_file_pack = os.path.join(target_folder, "f0data_crepe/MAST_melody_f0.zip")
    shutil.unpack_archive(f0_file_pack, audio_feature_files_folder)
    source_folder = audio_feature_files_folder+"MAST_melody_f0/" 
    move_all_files(source_folder, audio_feature_files_folder)
    
    # Inform user for the long data processing step
    if mode =='renameM4aToWav':
        print('!!!Attention, m4a to .wav conversion short-circuited, .wav files are obtained by just renaming .m4a')
        print('To get real .wav files, set mode to "convertM4aToWav" while calling download_melody_data_from_zenodo')
    elif mode =='convertM4aToWav':
        print('Converting all m4a files to .wav ....this will take a while')
    
    if mode =='convertM4aToWav' or mode == 'renameM4aToWav':
        for root, dirs, files in os.walk(audio_feature_files_folder):
            for filename in files:
                if '.m4a' in filename: # annotation files starts with 'report'
                    m4a_filename = os.path.join(audio_feature_files_folder, filename)
                    wav_filename = os.path.join(audio_feature_files_folder, filename.replace('.m4a', '.wav'))
                    if not os.path.exists(wav_filename):
                        if mode =='convertM4aToWav':
                            # Convert to .wav and delete m4a file
                            # IMPORTANT: if you fail to install the tools (ffmpeg) that reads and exports
                            # .wav files, the next lines may end up producing errors
                            # Then, you could simply rename .m4a files to .wav files setting mode to 'renameM4aToWav'
                            # if you are happy with using the features (f0-series and chroma) already provided
                            track = AudioSegment.from_file(m4a_filename,  format= 'm4a')
                            track.export(wav_filename, format='wav')
                            os.remove(m4a_filename)
                        else:
                            os.rename(m4a_filename, wav_filename)
    else:
        print('Mode error in calling download_melody_data_from_zenodo')
                        
    # Remove temporary folders
    for sub_folder in ["chroma/","audioFiles/","f0data_crepe/"]:
        sub_folder = target_folder + sub_folder
        if os.path.exists(sub_folder):
            shutil.rmtree(sub_folder, ignore_errors=True)

def main():
    target_folder = '../../data/'
    # Mode selection as below short-circuits audio format conversion, 
    # to get real .wav files, set it to 'convertM4aToWav'
    mode ='renameM4aToWav' 
    download_melody_data_from_zenodo(target_folder, mode=mode)

if __name__ == "__main__":
    main()