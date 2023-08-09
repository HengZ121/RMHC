import os
import cv2
import torch
import stattools
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.fft import fft, ifft
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler

# Location of fMRI data
fmri_path = r'C:\Users\Heng\Desktop\interpol-B-20230503T012138Z-004\interpol-B'
# Location of Psychology data
psy_path = r'C:\Users\Heng\Desktop\Raw_csv_Psychopy-20230505T025025Z-001\Raw_csv_Psychopy'

task_dict = {
    "DMSRun1" : "DMS",
    "DMSRun2" : "DMS",
    "NBackRun2" : 1,
    "VSTMRun1" : 2,
}

cortex_dict = {
    "DLPFC_left" : 0, 
    "DLPFC_right": 0, 
    "MOTOR1_left": 1,
    "MOTOR1_right":1,
    "VISUAL1_left":2, 
    "VISUAL1_right":2,
    "Hippo_left"  : 3,
    "Hippo_right" : 3
}

class Dataset():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self):
        # Parameters

        # Helper Attributes
        self.ids  = []
        self.nb_voxels = 0
        self.height = 0
        self.width = 0

        # Features (ACWs)
        self.features  = []
        # Labels (Abstracted from fMRI data)
        self.labels    = []

        # Helper Variables
        self.scaler = MinMaxScaler()

        print("Loading Labels and Features")
        for filename in tqdm(os.listdir(fmri_path)):
            # checking if it is a file
            task = task_dict[filename.split('_')[1]]

            if (task == "DMS"):

                # Read brain ACW0s of subject (labels)
               
                df = pd.read_csv(os.path.join(fmri_path, filename), sep=",", header=None)
                acw_df = []
                for index, row in df.iterrows():
                    acffunc = acw(row)
                    acw_df.append(acffunc)
                self.features.append(acw_df[0:250])
                name = filename.split('_')
                label = [0 for x in range(4)]
                label[cortex_dict[name[3] + '_' + name[4]]] = 1
                self.labels.append(label)

                if cortex_dict[name[3] + '_' + name[4]] == 3: ## Replicate minor class
                    self.features.append(acw_df[0:250])
                    self.labels.append(label)
    
                
        self.height = len(self.features[0])
        self.width = len(self.features[0][0])
        self.nb_voxels = len(self.features[0])
        print("All Data Loaded")

                # psd_df = []
                # # Transfer voxels' activation values by time to by frequency (Power Spectral Density)
                # for index, row in df.iterrows():
                #     (f, S) = scipy.signal.periodogram(np.array(row), fs, scaling='density')
                #     psd_df.append(S)
                
                # # Singal Strength Value -> Change in Singal Strength Value
                # for S in psd_df:
                #     ave_signal_strength = np.mean(S)
                #     for s in S:
                #         s = (s - ave_signal_strength) if s > ave_signal_strength else (ave_signal_strength - s)
                
                # psd_df = np.array(psd_df)

                # # Calculate the average singal strength of all voxel at a time point
                # psd_mean = np.mean(psd_df, axis=0)
                # self.psd.append(psd_mean)

                # N = len(df.columns)
                # fft_df = []
                # # Transfer voxels' activation values by time to by frequency (Fast Fourier Transformation)
                # for index, row in df.iterrows():
                #     # Do Fourier Transform to Transfer Row from time domain to frequency domain
                #     fft_row = fft(np.array(row))
                #     fft_row = ifft(fft_row).real.tolist()
                #     fft_freq_row = np.fft.fftfreq(N, d=1.111)[:N//2]
                #     fft_df.append(fft_row[:N//2])


                # Visualize ACW and save image
                # plt.imshow(np.array(acw_df, dtype=float), cmap='Greys', aspect='auto')
                # plt.xlabel("Frequency")
                # plt.ylabel("Voxels")
                # plt.savefig(os.path.join(r'C:\Users\Heng\Desktop\spectrograms', filename.replace('.1D','')+'.png'), bbox_inches='tight')
                # plt.close("all")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: sentence vector and label
        '''
        return torch.tensor(self.features[index],dtype=torch.float), torch.tensor(self.labels[index],dtype=torch.float)
    
#% ACW
def acw(ts, n_lag=None, fast=True):
    acffunc = acf(ts, nlags=n_lag, qstat=False, alpha=None, fft=fast)
    # acw5 = (2 * np.argmax(acffunc <= 0.5) - 1)
    # acw0 = (2 * np.argmax(acffunc <= 0) - 1)
    return acffunc


# #% PLE
# def ple(data,nperseg, noverlap,axis):
#     f,pxx = signal.welch(data, fs=fs, window='hann', nperseg =nperseg , noverlap=noverlap, scaling='spectrum', average='mean', axis = axis)
#     #log_f = np.log(f)
#     #log_p = np.log(pxx)
#     ple = -np.polyfit(np.log(f[1:]),np.log(pxx[1:]),1)[0]
#     return ple, f, pxx


#%MF
def mf(freq, psd):
    """
    Calculate the median frequency of a PSD
    Adapted from Mehrshad Golesorkhi's Neuro-Helper
    """
    cum_sum = np.cumsum(psd)
    medfreq = freq[np.argmax(cum_sum >= (cum_sum[-1] / 2))]
    return medfreq

#% LZC
def lzc_norm_fact(ts):
    """
    The default way of calculating LZC normalization factor for a time series
    :param ts: a time series
    :return: normalization factor
    """
    return len(ts) / np.log2(len(ts))


def lzc(ts, norm_factor=None):
    """
    Calculates lempel-ziv complexity of a single time series.
    :param ts: a time-series: nx1
    :param norm_factor: the normalization factor. If none, the output will not be normalized
    :return: the lempel-ziv complexity
    """
    bin_ts = np.char.mod('%i', ts >= np.median(ts))
    value = lempel_ziv_complexity("".join(bin_ts))
    if norm_factor:
        value /= norm_factor
    return value


def lempel_ziv_complexity(sequence):
    sub_strings = set()

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind: ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)

def pad_features(features):
    padded_features = []
    max_length = 0
    for cortex in features:
        if len(cortex) > max_length:
            max_length = len(cortex)
    for cortex in features:
        padding = max_length - len(cortex)
        padded_features.append([0 for x in range(padding//2)] + cortex + [0 for x in range(padding - padding//2)])
    return padded_features


