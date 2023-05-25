In this branch,

1. transferred fMRI data to frequency-based matrix (voxels versus frequency) by calculating power spectral density (PSD).
2. calculated the average signal strength of voxels at each frequency range
3. visualized the result and found that the average signal strengths of voxels converge from about 75 Hz (e.g., figure 1)

Therefore, I am thinking of creating labels based on the signal strength of voxels from the frequency ranges of greater than 75 Hz

![Figure 1](https://github.com/HengZ121/RMHC/blob/data-v2/figure1.png)

replace the paths in code to fMRI data folder and raw psychology data folder, and run <python3 classifier.py> to review