import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.fft import fft, ifft
from torchvision import transforms

# Location of fMRI data
fmri_path = r'C:\Users\Heng\Desktop\interpol-B-20230503T012138Z-004\interpol-B'
# Location of Psychology data
psy_path = r'C:\Users\Heng\Desktop\Raw_csv_Psychopy-20230505T025025Z-001\Raw_csv_Psychopy'

class FMRIData():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self, task, area):
        self.task = task
        self.area = area
        self.psd  = []
        self.ids  = []
        self.run  = []

        # features
        self.features  = []
        # labels are binary
        self.labels    = []

        print("Loading Labels")
        for filename in tqdm(os.listdir(fmri_path)):
            # checking if it is a file
            if (task in filename) and (area in filename):
                df = pd.read_csv(filename, sep=",", header=None)
                fs = 0.9
                psd_df = []
                for index, row in df.iterrows():

                    (f, S) = scipy.signal.periodogram(np.array(row), fs, scaling='density')

                    psd_df.append(S)
                
                psd_df = np.array(psd_df)
                psd_mean = np.mean(psd_df, axis=0)

                # N = len(df.columns)
                # fft_df = []
                # for index, row in df.iterrows():
                #     # Do Fourier Transform to Transfer Row from time domain to frequency domain
                #     fft_row = fft(np.array(row))
                #     fft_row = ifft(fft_row).real.tolist()
                #     fft_freq_row = np.fft.fftfreq(N, d=1.111)[:N//2]
                #     fft_df.append(fft_row[:N//2])
                # self.images.append(fft_df)


                # extent = [0,0,0,0]
                # plt.imshow(np.array(psd_df, dtype=float), cmap='Greys', vmin = 0, vmax= 7000, aspect='auto')
                # plt.xlabel("Frequency")
                # plt.ylabel("Voxels")
                # plt.savefig(os.path.join(r'C:\Users\Heng\Desktop\spectrograms', filename.replace('.1D','')+'.png'), bbox_inches='tight')
                # plt.close("all")

                self.psd.append(psd_mean)
                self.ids.append(filename.split('_')[2])
                self.run.append(1 if '1' in filename.split('_')[1] else 2)
        median_of_participants = np.median(self.psd, axis=0)
        for p in self.psd:
            self.labels.append(0 if np.sum(p - median_of_participants)< 0 else 1)
        print("Labels Loaded")

        print("Loading Features")
        for id in tqdm(self.ids):
            os.path.join(psy_path, )
            df = pd.read_csv(psy_path, sep=",", header=None)
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: sentence vector and label
        '''

        return self.descriptions[index], self.labels[index]

    def getImgShape(self):
        return len(self.images[0]), len(self.images[0][0]) 

