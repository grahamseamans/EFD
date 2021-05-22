# -*- coding: utf-6 -*-
"""
Created on Fri Apr 23 09:02:20 2021

@author: user
"""
import numpy as np
import h5py
import os
from PIL import Image
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numba import jit

# from scipy import signal
# Snapshot SVD densoising, remove background "noise"


def svd_decomp(data):
    U, S, VT = np.linalg.svd(data)
    S = np.diag(S)
    return U, S, VT



def pod(data):
    cov = np.cov(data)
    U, S, VT = np.linalg.svd(cov)
    return U, S, VT


def matrix_to_image(a):
    return (255 * (a - np.min(a)) / np.ptp(a)).astype(int)


matlab_save_location = r"C:\code_playbin\daniel_efd\data\tir\july9_2200_first"
image_save_location = r"C:\code_playbin\daniel_efd\image_files"

files = []
for file in os.listdir(matlab_save_location):
    files.append(file)

os.chdir(matlab_save_location)

matlab_list = []
for file in files:
    with h5py.File(file) as f:
        matlab_list.append(f["data"][:])

video = np.concatenate(matlab_list, axis=0)
video = np.flip(video, axis=1)
video_pixel_mean = np.mean(video, axis=0)

reshaped_video = []

for i in range(len(video)):
    reshaped_video.append(video[i].reshape(1024 * 768))

del video
del matlab_list

reshaped_video = np.stack(reshaped_video, axis=1)
U, S, VT = np.linalg.svd(reshaped_video, full_matrices=False)
S = np.diag(S)

print(U)


@jit(nopython=True)
def high_low_svd_filter(U,S,VT, highest_kept_mode, lowest_kept_mode):
    return (
        U[:, highest_kept_mode:lowest_kept_mode]
        @ S[highest_kept_mode:lowest_kept_mode, highest_kept_mode:lowest_kept_mode]
        @ VT[highest_kept_mode:lowest_kept_mode, :]
    )

def svd_and_filter(tensor, highest_kept_mode, lowest_kept_mode):
    U, S, VT = svd_decomp(tensor) 
    return high_low_svd_filter(U,S,VT, highest_kept_mode=highest_kept_mode, lowest_kept_mode=lowest_kept_mode)

video = []
lowest_kept_mode = 400
highest_kept_mode = 0
lowest_background_mode = 12
highest_background_mode = 0

n = 2

bandpass = high_low_svd_filter(
    U,S,VT, highest_kept_mode=0, lowest_kept_mode=400 
)

print(U)

background = high_low_svd_filter(
    U,S,VT,  highest_kept_mode=0, lowest_kept_mode=12 
)


def get_mode(u, basis):
    return u[:, basis]


num_basis_vectors = 20

axes = []
fig = plt.figure()

for i in range(num_basis_vectors):
    bimage = get_mode(U, i).reshape(1024, 768)
    axes.append(fig.add_subplot(3, 7, i + 1))
    subplot_title = "Basis " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)
plt.show()


for i in range(reshaped_video.shape[1]):
    x = reshaped_video[:, i].reshape(1024, 768) - background[:, i].reshape(1024, 768)
    video.append(x)


def svd_of_image(video, lowest_kept_mode):
    twice_svd_video = []
    for frame in video:
        bandpass = svd_and_filter(
            frame, highest_kept_mode=0, lowest_kept_mode=lowest_kept_mode
        )
        fill = np.mean(frame)
        bandpass = np.where(np.abs(bandpass) < 0.3, bandpass, fill)
        twice_svd_video.append(bandpass)
    return twice_svd_video


twice_svd_video = svd_of_image(video, lowest_kept_mode=400)

alph = list(string.ascii_lowercase)

aa = []
for i in alph:
    for j in alph:
        aa.append(i + j)

os.chdir(image_save_location)
for i in range(len(twice_svd_video)):
    d1 = matrix_to_image(twice_svd_video[i])
    im = Image.fromarray(d1.astype(np.uint8))
    a = aa[i // 10]
    n = str(i % 10)
    imname = "pod" + a + n + ".bmp"
    im.save(imname)
