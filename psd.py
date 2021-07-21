# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:25:03 2021

@author: user
"""
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import norm, pearsonr
import seaborn as sns
import scipy.signal as signal

rng = np.random.RandomState(42)
pixn = 100

# Process from matlab files to TIV by file
matout = r"C:\Users\PIV\Daniel\1hz1630"  # Location of MAT input files
nfiles = 5  # number of files to use at a time
files = []
for file in os.listdir(matout):
    files.append(file)

os.chdir(matout)
files = []
for file in os.listdir(matout):
    files.append(file)


count = 0
for num in range(len(files) // nfiles):
    mfiles = files[num * nfiles : (num + 1) * nfiles]
    os.chdir(matout)
    count += 1
    tirl = []
    for file in mfiles:
        with h5py.File(file, "r") as f:
            tirl.append(f["data"][:])
    tir = np.concatenate(tirl, axis=0)
    tirm = np.mean(tir, axis=0)
    tir -= tirm

    secs = np.arange(tir.shape[0])
    tir = np.flip(tir, axis=1)
    tirl2 = []

    for i in range(len(tir)):
        tirl2.append(tir[i].reshape(1024 * 768))

    del tir
    tirl2 = np.stack(tirl2, axis=1)
    U, S, VT = np.linalg.svd(tirl2, full_matrices=False)
    sl = S[:20]
    S = np.diag(S)

    moden = 0
    rfl = []
    rwl = []
    for n in range(4):
        moden += 1
        if moden == 1:
            b = 1
            l = 0
            tir = U[:, l:b] @ S[l:b, l:b] @ VT[l:b, :]
            tir = tir.reshape(1024, 768, 400)
            tmean = np.mean(tir, axis=2)
            if 2 == 3:
                plt.subplots()
                ax = sns.displot(tmean.flatten(), bins=100, kde=True)
                title = "batch " + str(count) + " mode " + str(moden)
                plt.title(title)
                filename = r"C:\Users\PIV\Daniel\psd\histogram " + title + ".png"
            # plt.savefig(filename)
            cold = np.argwhere(tmean < np.mean(tmean))
            hot = np.argwhere(tmean > np.mean(tmean))
            ranp = np.argwhere(tmean != np.nan)
            ranp = ranp[rng.choice(len(ranp), pixn, replace=False), :]
            hot = hot[rng.choice(len(hot), pixn, replace=False), :]
            cold = cold[rng.choice(len(cold), pixn, replace=False), :]

            dpl = []
            for i in range(tir.shape[2]):
                pix = []
                for j in hot:
                    pix.append(tir[j[0], j[1], i])
                dpl.append(pix)

            dp = np.stack(dpl)

            lpl = []
            for i in range(tir.shape[2]):
                pix = []
                for j in cold:
                    pix.append(tir[j[0], j[1], i])
                lpl.append(pix)

            lp = np.stack(lpl)

            rpl = []
            for i in range(tir.shape[2]):
                pix = []
                for j in ranp:
                    pix.append(tir[j[0], j[1], i])
                rpl.append(pix)

            rp = np.stack(rpl)

        else:

            b = n + 1
            m = n
            l = 0
            tir = U[:, l:b] @ S[l:b, l:b] @ VT[l:b, :]
            subg = U[:, l:m] @ S[l:m, l:m] @ VT[l:m, :]
            tir -= subg
            tir = tir.reshape(1024, 768, tir.shape[1])
            tmean = np.mean(tir, axis=2)

            if 2 == 3:
                plt.subplots()
                ax = sns.displot(tmean.flatten(), bins=100, kde=True)
                title = "batch " + str(count) + " mode " + str(moden)
                plt.title(title)
                filename = r"C:\Users\PIV\Daniel\psd\histogram " + title + ".png"
            # plt.savefig(filename)

            cold = np.argwhere(tmean < np.mean(tmean))
            hot = np.argwhere(tmean > np.mean(tmean))
            ranp = np.argwhere(tmean != np.nan)
            ranp = ranp[rng.choice(len(ranp), pixn, replace=False), :]
            hot = hot[rng.choice(len(hot), pixn, replace=False), :]
            cold = cold[rng.choice(len(cold), pixn, replace=False), :]
            for i in range(tir.shape[2]):
                pix = []
                for j in hot:
                    pix.append(tir[j[0], j[1], i])
                dpl.append(pix)

            dp = np.stack(dpl)

            lpl = []
            for i in range(tir.shape[2]):
                pix = []
                for j in cold:
                    pix.append(tir[j[0], j[1], i])
                lpl.append(pix)

            lp = np.stack(lpl)

            rpl = []
            for i in range(tir.shape[2]):
                pix = []
                for j in ranp:
                    pix.append(tir[j[0], j[1], i])
                rpl.append(pix)

            rp = np.stack(rpl)

        """
        lstd = np.std(lp,axis=1)
        dstd = np.std(dp,axis=1)
        rstd = np.std(rp,axis=1)
        
        lmean = np.mean(lp,axis=1)
        dmean = np.mean(dp,axis=1)
        rmean = np.mean(rp,axis=1)
        
        
        plt.subplots()
        plt.plot(secs,lstd)
        plt.plot(secs,dstd)
        plt.legend(['Cold','Hot'])
        
        plt.subplots()
        plt.plot(secs,lmean)
        plt.plot(secs,dmean)
        plt.legend(['Cold','Hot'])
        """

        ld = detrend(lp, axis=0)
        dd = detrend(dp, axis=0)
        rd = detrend(rp, axis=0)

        """
        #Uses non-detrended light and dark, results are similar but more dramatic
        ld = lp
        dd = dp
        """
        # Compute PSD on Dark and Light Patches

        lpw = []
        lf = []
        for i in range(lp.shape[1]):
            f, P = signal.welch(signal.detrend(lp[:, i], axis=0))
            lpw.append(P)
            lf.append(f)

        lpw = np.stack(lpw)
        lpw = np.mean(lpw, 0)
        lf = lf[0]

        dpw = []
        df = []
        for i in range(dp.shape[1]):
            f, P = signal.welch(signal.detrend(dp[:, i], axis=0))
            dpw.append(P)
            df.append(f)

        dpw = np.stack(dpw)
        dpw = np.mean(dpw, 0)
        df = df[0]

        rpw = []
        rf = []
        for i in range(rp.shape[1]):
            f, P = signal.welch(signal.detrend(rp[:, i], axis=0))
            rpw.append(P)
            rf.append(f)

        rpw = np.stack(rpw)
        rpw = np.mean(rpw, 0)
        rf = rf[0]
        rfl.append(rf)
        rwl.append(rpw)

    plt.subplots()
    leg = []
    con = 0
    for f, p in zip(rfl, rwl):
        # plt.loglog(df,dpw)
        # plt.loglog(lf,lpw)
        plt.semilogx(f, p)
        leg.append("Mode " + str(con + 1) + " S value " + str(int(sl[con])))
        con += 1
        # plt.title('Mean of PSD of Hot, Cold and All Patches Batch '+ str(count)+ ' Mode ' + str(moden))

    title = "Semilogx 1630 Mean of PSD of Random Pixels Batch " + str(count)
    filename = r"C:\Users\PIV\Daniel\psd//" + title + ".png"
    plt.title(title)
    plt.legend(leg)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [T**2/Hz]")

    # plt.legend(['Hot Patches','Cold Patches','All Patches'])
    slope = -5 / 3
    dx = 0.1
    x0 = 0.004
    y0 = 10
    x1, y1 = [x0, x0 + dx], [x0 ** slope / 2000, (x0 + dx) ** slope / 2000]
    # plt.plot(x1, y1)
    # plt.text(0.02,1,'-5/3 Slope',bbox=dict(facecolor='none', edgecolor='black'))
    plt.savefig(filename)
    plt.show()
