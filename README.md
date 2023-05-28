In this branch,

1. transferred fMRI data to frequency-based matrix (voxels versus frequency) by calculating power spectral density (PSD).
2. calculated the average signal strength of voxels at each frequency range
3. visualized the result and found that the average signal strengths of voxels converge from about 75 Hz (e.g., figure 1)
4. calculated the average of (2.) of all subjects
5. generate binary label (0 if the sum of distance from one subject's average voxels strengths is lower than the averages of all subjects in (3.) over the frequency domain, else 1)

Therefore, there will be two labels: below the average, over the average; the features we use to feed the machine learning model (logic regression, classifier. in this branch) are below columns in raw psy data:


round_feature, round1_res_mean, round2_res_mean, round3_res_mean, round4_res_mean, round5_res_mean,
               round1_corr, round2_corr, round3_corr, round4_corr, round5_corr,
               round1_res_corr, round2_res_corr, round3_res_corr, round4_res_corr, round5_res_corr,
               round1_res_rt, round2_res_rt, round3_res_rt, round4_res_rt, round5_res_rt,
               round1_res_key, round2_res_key, round3_res_key, round4_res_key, round5_res_key
               
![Figure 1](https://github.com/HengZ121/RMHC/blob/data-v2/figure1.png)

replace the paths in code to fMRI data folder and raw psychology data folder, and run <python3 classifier.py> to review





