# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:27:19 2021

@author: user
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from openpiv import tools, pyprocess, scaling, validation, filters
import os
import matplotlib.animation as animation
from PIL import Image
from sklearn.metrics import mean_absolute_error


# matlab_save_location = r"C:\code_playbin\daniel_efd\data\tir\july9_2200_first"
# image_save_location = r"C:\code_playbin\daniel_efd\image_files"

# lems = r"C:\Users\user\mlems"
lems = r"C:\code_playbin\daniel_efd\data\alllems"
imdir = r"C:\code_playbin\daniel_efd\image_files"
# imdir = r"C:\Users\user\im\podsvd"
# base = r"C:\\Users\user"
base = r"C:\code_playbin\daniel_efd"

os.chdir(lems)
sl = []
dl = []

sens = []
for f in os.listdir(lems):
    if f[4] == "T" or f[4] == "K":
        lin = []
        lemsf = pd.read_csv(f)
        sens.append(f[4])
        for i in lemsf.index:
            y = lemsf["Year"][i]
            m = lemsf["Month"][i]
            d = lemsf["Date"][i]
            h = lemsf["Hour"][i]
            mi = lemsf["Minute"][i]
            s = lemsf["Second"][i]
            ts = pd.Timestamp(year=y, month=m, day=d, hour=h, minute=mi, second=s)
            lin.append(ts)
        direc = pd.Series(data=lemsf["Sonic_Dir"].values, index=lin)
        spd = pd.Series(data=lemsf["Sonic_Spd"].values, index=lin)
        dl.append(direc)
        sl.append(spd)
    else:
        pass

start = "2019-07-09-21:55:00"
stop = "2019-07-09-22:1:19"

# start = '2019-07-09-16:27:20'
# stop = '2019-07-09-16:55:40'
sdf = pd.concat(sl, axis=1)
ddf = pd.concat(dl, axis=1)

sdf = sdf.loc[(sdf.index >= start) & (sdf.index < stop)]
ddf = ddf.loc[(ddf.index >= start) & (ddf.index < stop)]


sonicmag = sdf.mean(axis=1)  # target variables (mag,dir)
sonicdir = ddf.mean(axis=1)

speeddiff = np.abs(sdf[0] - sdf[1])
dirdiff = np.abs(ddf[0] - ddf[1])


def filter_perc(arr):
    return arr[(arr < np.percentile(arr, 75)) & (arr > np.percentile(arr, 25))]


files = os.listdir(imdir)
os.chdir(imdir)

ul = []
vl = []
for i in range(len(files) - 1):
    frame_a = tools.imread(files[i], 0)
    frame_b = tools.imread(files[i + 1], 0)

    ws = 106
    ol = 0
    dt = 1
    ss = 206
    sig2noisemethod = "peak2peak"
    scalingfactor = 0.1

    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=ws,
        overlap=ol,
        dt=dt,
        search_area_size=ss,
        sig2noise_method=sig2noisemethod,
    )
    u, v, mask1 = validation.sig2noise_val(u, v, sig2noise, threshold=0.6)
    u, v, mask2 = validation.global_val(u, v, (-1, 1), (-1, 1))
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape, search_area_size=ss, overlap=ol
    )
    u = np.where(u < 1, u, np.nan)
    u = np.where(u > -1, u, np.nan)
    v = np.where(v < 1, v, np.nan)
    v = np.where(v > -1, v, np.nan)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=scalingfactor)
    u = u[np.logical_not(np.isnan(u))]
    v = v[np.logical_not(np.isnan(v))]
    if len(u) > 0 and len(v) > 0:
        u = filter_perc(u.flatten())
        v = filter_perc(v.flatten())
    else:
        u = 0
        v = 0

    ul.append(np.nanmean(u))
    vl.append(np.nanmean(v))

angle = []
for i, j in zip(ul, vl):
    angle.append(
        (np.degrees(np.arctan(i / j)) + 65) % 360  # reorients vector into real space
    )  # orientation = 295, vec to dir = 180
mag = []
for i, j in zip(ul, vl):
    mag.append(np.sqrt(i ** 2 + j ** 2))
"""
plt.subplots()
plt.scatter(np.arange(0,len(angle),1),angle)

plt.subplots()
plt.scatter(np.arange(0,len(mag),1),mag)
"""


lx = [0]
ly = np.arange(0, len(angle), 1)

xx, yy = np.meshgrid(lx, ly)


plt.subplots()
plt.quiver(yy, xx, ul, vl)
plt.title("SVD TIV")


tivdir = []
for i in range(len(angle) // 10 - 1):
    tivdir.append(np.nanmedian(angle[i * 10 : (i + 1) * 10]))

dirdf = pd.DataFrame(sonicdir, columns=["LEMS"])
dirdf["TIV"] = tivdir
dirdf.plot()

tivmag = []
for i in range(len(mag) // 10 - 1):
    tivmag.append(np.nanmedian(mag[i * 10 : (i + 1) * 10]))

magdf = pd.DataFrame(sonicmag, columns=["LEMS"])
magdf["TIV"] = tivmag
magdf.plot()


print("MAE Direction", mean_absolute_error(sonicdir, tivdir))
print("MAE Magnitude", mean_absolute_error(sonicmag, tivmag))
