import os
import cv2
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.fft import fft, ifft
from sklearn.preprocessing import StandardScaler

# Location of fMRI data
fmri_path = r'C:\Users\Heng\Desktop\interpol-B-20230503T012138Z-004\interpol-B'
# Location of Psychology data
psy_path = r'C:\Users\Heng\Desktop\Raw_csv_Psychopy-20230505T025025Z-001\Raw_csv_Psychopy'

class Dataset():

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

        scaler = StandardScaler()

        print("Loading ", self.area," Labels")
        for filename in tqdm(os.listdir(fmri_path)):
            # checking if it is a file
            if (task in filename) and (area in filename):
                df = pd.read_csv(os.path.join(fmri_path, filename), sep=",", header=None)
                fs = 0.9
                psd_df = []
                for index, row in df.iterrows():

                    (f, S) = scipy.signal.periodogram(np.array(row), fs, scaling='density')

                    psd_df.append(S)
                
                for S in psd_df:
                    ave_signal_strength = np.mean(S)
                    for s in S:
                        s = (s - ave_signal_strength) if s > ave_signal_strength else (ave_signal_strength - s)
                
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
                self.run.append('1' if '1' in filename.split('_')[1] else '2')

        print("Loading Features")
        labels_to_be_deleted = []
        for index, id in tqdm(enumerate(self.ids)):
            path_of_raw_data = os.path.join(psy_path, id)
            path_of_raw_data = os.path.join(path_of_raw_data, os.listdir(path_of_raw_data)[0])
            found = False
            for filename2 in os.listdir(path_of_raw_data):
                if self.run[index] in filename2 and self.task in filename2 and 'MRI' in filename2:
                    found = True
                    df = pd.read_csv(os.path.join(path_of_raw_data, filename2), encoding= 'unicode_escape', on_bad_lines='skip')
                    df = df.replace(np.nan,-10)
                    df = df.replace("None", 0)
                    round_feature   = df['Load'].max()
                    round1_res_mean = (df['StartTime'][ 5] - df['StartTime'][ 0])/5
                    round2_res_mean = (df['StartTime'][12] - df['StartTime'][ 7])/5
                    round3_res_mean = (df['StartTime'][19] - df['StartTime'][14])/5
                    round4_res_mean = (df['StartTime'][26] - df['StartTime'][21])/5
                    round5_res_mean = (df['StartTime'][33] - df['StartTime'][28])/5
                    round1_corr    = sum(df['corr'][ 0: 5].values.tolist())
                    round2_corr    = sum(df['corr'][ 7:12].values.tolist())
                    round3_corr    = sum(df['corr'][14:19].values.tolist())
                    round4_corr    = sum(df['corr'][21:26].values.tolist())
                    round5_corr    = sum(df['corr'][28:33].values.tolist())
                    round1_res_corr    = sum(df['resp.corr'][ 0: 5].values.tolist())
                    round2_res_corr    = sum(df['resp.corr'][ 7:12].values.tolist())
                    round3_res_corr    = sum(df['resp.corr'][14:19].values.tolist())
                    round4_res_corr    = sum(df['resp.corr'][21:26].values.tolist())
                    round5_res_corr    = sum(df['resp.corr'][28:33].values.tolist())
                    round1_res_rt      = sum(df['resp.rt'][ 0: 5].values.tolist())/5
                    round2_res_rt      = sum(df['resp.rt'][ 7:12].values.tolist())/5
                    round3_res_rt      = sum(df['resp.rt'][14:19].values.tolist())/5
                    round4_res_rt      = sum(df['resp.rt'][21:26].values.tolist())/5
                    round5_res_rt      = sum(df['resp.rt'][28:33].values.tolist())/5
                    round1_res_key     = sum(pd.to_numeric(df['resp.keys'])[ 0: 5].values.tolist())
                    round2_res_key     = sum(pd.to_numeric(df['resp.keys'])[ 7:12].values.tolist())
                    round3_res_key     = sum(pd.to_numeric(df['resp.keys'])[14:19].values.tolist())
                    round4_res_key     = sum(pd.to_numeric(df['resp.keys'])[21:26].values.tolist())
                    round5_res_key     = sum(pd.to_numeric(df['resp.keys'])[28:33].values.tolist())
                    self.features.append([round_feature, round1_res_mean, round2_res_mean, round3_res_mean, round4_res_mean, round5_res_mean,
                                         round1_corr, round2_corr, round3_corr, round4_corr, round5_corr,
                                         round1_res_corr, round2_res_corr, round3_res_corr, round4_res_corr, round5_res_corr,
                                         round1_res_rt, round2_res_rt, round3_res_rt, round4_res_rt, round5_res_rt,
                                         round1_res_key, round2_res_key, round3_res_key, round4_res_key, round5_res_key])
                    break
            if not found:
                # no corresponding raw psy data, remove the label
                labels_to_be_deleted.append(index)
        labels_to_be_deleted.reverse()
        for index in labels_to_be_deleted:
                del self.ids[index]
                del self.run[index]
                del self.psd[index]
        x = []
        mean_of_participants = np.mean(self.psd, axis=0)
        for p in self.psd:
            self.labels.append(0 if np.sum(p - mean_of_participants)< 0 else 1)
            y = p - mean_of_participants
            x = np.array([i for i in range(len(y))])
            plt.scatter(x, y, color="red")
            plt.show()
            # x.append(np.sum(p - mean_of_participants))
        print("Labels Loaded")
        # x = np.sort(np.array(x))
        # y = np.array([i for i in range(len(x))])
        # print("Labels Loaded")
        # plt.title(self.area)
        # plt.scatter(y, x, color="red")
        # plt.show()
        scaler.fit(self.features)
        self.features = scaler.transform(self.features)

        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: sentence vector and label
        '''
        return self.features[index], self.labels[index]
    

