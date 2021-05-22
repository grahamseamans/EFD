# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:02:20 2021

@author: user
"""
import numpy as np
import h5py
import os
from PIL import Image
import string
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# SVD denoising for each image


def nn(a):
    return (255 * (a - np.min(a)) / np.ptp(a)).astype(int)


# matout = r'C:\Users\user\five22'
# imout = r'C:\Users\user\im\test22'

matout = r"C:\Users\user\five22"
imout = r"C:\Users\user\im\test22"

files = []
for file in os.listdir(matout):
    files.append(file)

os.chdir(matout)

tirl = []
for file in files:
    with h5py.File(file) as f:
        tirl.append(f["data"][:])

tir = np.concatenate(tirl, axis=0)
tir = np.flip(tir, axis=1)
tir = signal.detrend(tir, axis=0)
tir = tir - np.mean(tir, axis=0)

vls = []
tirf = []

r = 400
for i in range(len(tir)):
    U, S, VT = np.linalg.svd(tir[i])
    S = np.diag(S)
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    fill = np.mean(tir[i])
    Xapprox = np.where(np.abs(Xapprox) < 0.2, Xapprox, fill)
    tirf.append(Xapprox)
    var_explained = np.round(S ** 2 / np.sum(S ** 2), decimals=3)
    vls.append(var_explained.diagonal())


vls = vls / 768

plt.subplots()
plt.semilogy(np.arange(768), vls)

# tirf = tirf-np.mean(tirf,axis=0)

# tirfg = gaussian_filter(tirf, sigma=0.5) #Applies gaussian filter, not very good
# tirf = tir-np.mean(tir) #Just uses the detrended fluctuation
alph = list(string.ascii_lowercase)

aa = []
for i in alph:
    for j in alph:
        aa.append(i + j)

os.chdir(imout)
for i in range(len(tirf)):
    d1 = nn(tirf[i])
    im = Image.fromarray(d1.astype(np.uint8))
    a = aa[i // 10]
    n = str(i % 10)
    imname = "pca" + a + n + ".bmp"
    im.save(imname)
