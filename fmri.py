import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.fft import fft, ifft
from torchvision import transforms

dir_path = r'C:\Users\Heng\Desktop\interpol-B-20230503T012138Z-004\interpol-B'

class FMRIData():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self, task, area):
        self.task = task
        self.area = area
        self.images = []
        self.images_descriptions = []
        print("Loading fMRI info.")
        for filename in tqdm(os.listdir(dir_path)):
            # checking if it is a file
            if (task in filename) and (area in filename):
                df = pd.read_csv(filename, sep=",", header=None)
                # N = len(df.columns)
                # fft_df = []

                # for index, row in df.iterrows():
                #     # Do Fourier Transform to Transfer Row from time domain to frequency domain
                #     fft_row = fft(np.array(row))
                #     fft_row = ifft(fft_row).real.tolist()
                #     fft_freq_row = np.fft.fftfreq(N, d=1.111)[:N//2]
                #     fft_df.append(fft_row[:N//2])
                
                # # extent = [0,0,0,0]
                # # extent[1] = fft_freq_row[len(fft_freq_row)-1]
                # # extent[2] =  len(fft_df) - 1
                # # plt.imshow(np.array(fft_df, dtype=float), extent=extent, cmap='Greys', vmin = -2500, vmax= 2500, aspect='auto')
                # # plt.xlabel("Frequency")
                # # plt.ylabel("Voxels")
                # # plt.savefig(os.path.join(r'C:\Users\Heng\Desktop\spectrograms', filename.replace('.1D','')+'.png'), bbox_inches='tight')
                # # plt.close("all")

                self.images.append(df.values.tolist())
                self.images_descriptions.append(filename)

        print("All fMRI data are Fourier transformed")

        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: sentence vector and label
        '''

        return torch.tensor(self.images[index]).float(), self.images_descriptions[index]

    def getImgShape(self):
        return len(self.images[0]), len(self.images[0][0]) 

