"""
This file defines the parameters/constants
"""


CREPE_F0_CONFIDENCE_LIMIT = 0.5 
MINIMUM_FREQUENCY = 30
MED_FILT_KERNEL_SIZE_F0 = 5

CHROMA_NFFT = 2048
CHROMA_NFRAMES = 128
CHROMA_NOTES_PER_OCTAVE = 24
CHROMA_TYPE = 'STFT'

F0IMG_NFRAMES = 128
F0IMG_FREQ_BINS = 72
F0_HOPLEN_SEC = 0.01 

CENTS_PER_OCTAVE = 1200.0
CENT_REF55HZ = 55.0

ALIGNED_F0_LEN = 512

diff_hist_bins = [0,  25,  50,  75, 100, 125, 150, 200, 1200]

feature_names = ['f0_dtw_norm_dist',
    'f0_dtw_len_mod_ref',
    'f0_dtw_len_mod_std',
    'f0_mean_diff',
    'f0_std_diff',
    'f0_last_10perc_mean_diff',
    'f0_diff_hist0',
    'f0_diff_hist1',
    'f0_diff_hist2',
    'f0_diff_hist3',
    'f0_diff_hist4',
    'f0_diff_hist5',
    'f0_diff_hist6',
    'f0_diff_hist7',
    'chrm_dtw_norm_dist',
    'chrm_dtw_len_mod_ref',
    'chrm_dtw_len_mod_std',
    'chrm_mean_diff',
    'chrm_std_diff',
    'chrm_last_10perc_mean_diff',
    'chrm_diff_hist0',
    'chrm_diff_hist1',
    'chrm_diff_hist2',
    'chrm_diff_hist3',
    'chrm_diff_hist4',
    'chrm_diff_hist5',
    'chrm_diff_hist6',
    'chrm_diff_hist7'] 





