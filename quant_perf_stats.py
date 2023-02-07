# %% quantify and find the optimal output on one FOV
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
import scipy.io
import numpy as np
import os, glob
import re

rd_path = r'D:\Luzhe_HIDEF\Data\Lung\Lung_Testset_size=512_prop=500_n=2\RH_M_chn=16_mae=3.0_ssim=1.0_lr=1.0e-05-1.0e-06_n=5_trial=mix_backbone'

# calculate rmse map
def get_metrics(mfile, patch=None):
    # patch: [x_lt, y_lt, x_rb, y_rb]
    # return metrics: amp. RMSE, phase RMSE, ECC
    mat_file = os.path.join(rd_path, mfile)
    tmp = scipy.io.loadmat(mat_file)

    if patch is not None:
        target = tmp['targetData'][patch[0]:patch[2],patch[1]:patch[3],:]
        output = tmp['outputData'][patch[0]:patch[2],patch[1]:patch[3],:]
    else:
        target = tmp['targetData']
        output = tmp['outputData']
    amp_rmse = np.sqrt(np.square(target[:,:,0]-output[:,:,0]).mean())
    ph_rmse = np.sqrt(np.square(target[:,:,1]-output[:,:,1]).mean())
    ecc = np.real(np.sum(target[:,:,0]*np.exp(1j*target[:,:,1]) * output[:,:,0]*np.exp(-1j*output[:,:,1]))) / np.sqrt(np.sum(np.square(target[:,:,0])) * np.sum(np.square(output[:,:,0])))
    return amp_rmse, ph_rmse, ecc

# %% quantitatively evaluate all test outputs
mfiles = glob.glob(os.path.join(rd_path, 'lung_MT_S2_R1_patch=03,03_*.mat'))
amp_rmse_list, ph_rmse_list, ecc_list = np.zeros([len(mfiles)]), np.zeros([len(mfiles)]), np.zeros([len(mfiles)])
# traverse mat files
for i in range(len(mfiles)):
    if i % 100 == 0:
        print("DEBUG: mat file %d" %i)
    mfile = os.path.join(rd_path, mfiles[i].split('\\')[-1])
    amp_rmse, ph_rmse, ecc = get_metrics(mfiles[i])
    # amp_rmse, ph_rmse, ecc = get_metrics(mfiles[i], [600,1112,800,1312])
    amp_rmse_list[i], ph_rmse_list[i], ecc_list[i] = amp_rmse, ph_rmse, ecc
# argmax to get the optimal output
print('The optimal output (Amp. RMSE) is: ', mfiles[np.argmin(amp_rmse_list)])
print('The optimal output (Phase RMSE) is: ', mfiles[np.argmin(ph_rmse_list)])
print('The optimal output (ECC) is: ', mfiles[np.argmax(ecc_list)])

# %% calculate mean SSIM and RMSE
amp_rmse_mean, amp_rmse_std = np.mean(amp_rmse_list), np.std(amp_rmse_list)
ph_rmse_mean, ph_rmse_std = np.mean(ph_rmse_list), np.std(ph_rmse_list)
ecc_mean, ecc_std = np.mean(ecc_list), np.std(ecc_list)

print('The amp. RMSE on all testing data is %.3f+/-%.3f' % (amp_rmse_mean, amp_rmse_std))
print('The phase RMSE on all testing data is %.3f+/-%.3f' % (ph_rmse_mean, ph_rmse_std))
print('The ECC on all testing data is %.4f+/-%.4f' % (ecc_mean, ecc_std))


# %%
