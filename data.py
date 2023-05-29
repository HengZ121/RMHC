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
from sklearn.preprocessing import StandardScaler

# Location of fMRI data
fmri_path = r'C:\Users\Heng\Desktop\interpol-B-20230503T012138Z-004\interpol-B'
# Location of Psychology data
psy_path = r'C:\Users\Heng\Desktop\Raw_csv_Psychopy-20230505T025025Z-001\Raw_csv_Psychopy'

label_dict = {
    "DMSRun1" : 0,
    "DMSRun2" : 0,
    "NBackRun2" : 1,
    "VSTMRun1" : 2,
}

lr_dict = {
    "MOTOR1_left" : 0.00045,
    "DLPFC_left"  : 0.0002,
    "DLPFC_right" : 0.0002,
    "VISUAL1_left" : 0.0002,
    "VISUAL1_right" : 0.0002,
}

epoch_dict = {
    "MOTOR1_left" : 70,
    "DLPFC_left" : 100,
    "DLPFC_right" : 100,
    "VISUAL1_left" : 100,
    "VISUAL1_right" : 100,
}

class Dataset():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self, area):
        # Parameters
        self.area = area

        # Helper Attributes
        self.ids  = []
        self.nb_voxels = 0
        self.height = 0
        self.width = 0
        self.learning_rate = lr_dict[area]
        self.epoch = epoch_dict[area]

        # Features (ACWs)
        self.features  = []
        # Labels (Abstracted from fMRI data)
        self.labels    = []

        print("Loading ", self.area," Labels")
        for filename in tqdm(os.listdir(fmri_path)):
            # checking if it is a file
            if (area in filename):
                df = pd.read_csv(os.path.join(fmri_path, filename), sep=",", header=None)
                fs = 0.9

                acw_df = []
                for index, row in df.iterrows():
                    acffunc = acw(row)
                    acw_df.append(acffunc)
                self.ids.append(filename.split('_')[2])
                label = [0,0,0]
                label[label_dict[filename.split('_')[1]]] = 1
                if (label_dict[filename.split('_')[1]] == 1):
                    self.labels.append(label)
                    self.features.append(acw_df)
                    self.labels.append(label)
                    self.features.append(acw_df)
                elif (label_dict[filename.split('_')[1]] == 2):
                    for _ in range(5):
                        self.labels.append(label)
                        self.features.append(acw_df)
                else:
                    self.labels.append(label)
                    self.features.append(acw_df)
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
    acw5 = (2 * np.argmax(acffunc <= 0.5) - 1)
    acw0 = (2 * np.argmax(acffunc <= 0) - 1)
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
    

