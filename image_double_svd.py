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


resample_factor = 5


@jit(nopython=True)
def svd_decomp(data, full_matricies=False):
    U, S, VT = np.linalg.svd(data, full_matricies)
    S = np.diag(S)
    return U, S, VT


@jit(nopython=True)
def pod(data):
    return svd_decomp(np.cov(data))


def matrix_to_image(a):
    return (255 * (a - np.min(a)) / np.ptp(a)).astype(int)


@jit(nopython=True)
def high_low_svd_filter(U, S, VT, highest_kept_mode, lowest_kept_mode):
    U = np.ascontiguousarray(U[:, highest_kept_mode:lowest_kept_mode])
    S = np.ascontiguousarray(
        S[highest_kept_mode:lowest_kept_mode, highest_kept_mode:lowest_kept_mode]
    )
    VT = np.ascontiguousarray(VT[highest_kept_mode:lowest_kept_mode, :])
    return U @ S @ VT


@jit(nopython=True)
def svd_and_filter(tensor, highest_kept_mode, lowest_kept_mode):
    U, S, VT = svd_decomp(tensor)
    return high_low_svd_filter(
        U, S, VT, highest_kept_mode=highest_kept_mode, lowest_kept_mode=lowest_kept_mode
    )


def get_mode(u, basis):
    return u[:, basis]


def vect_to_frame(vect):
    return vect.reshape(1024, 768)


@jit(nopython=True)
def resample_matrix(matrix):
    return matrix[::resample_factor, ::resample_factor]


# read in data
matlab_save_location = os.path.join(os.getcwd(), "data", "tir", "july9_2200_first")
image_save_location = os.path.join(os.getcwd(), "data", "images")

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

vectorized_frame_video = []
for i in range(len(video)):
    vectorized_frame_video.append(video[i].reshape(1024 * 768))

del video
del matlab_list

# svd across video
vectorized_frame_video = np.stack(vectorized_frame_video, axis=1)


U, S, VT = svd_decomp(vectorized_frame_video, full_matricies=False)

bandpass = high_low_svd_filter(U, S, VT, highest_kept_mode=0, lowest_kept_mode=400)
background = high_low_svd_filter(U, S, VT, highest_kept_mode=0, lowest_kept_mode=12)

# num_basis_vectors = 20
# axes = []
# fig = plt.figure()
# for i in range(num_basis_vectors):
#     bimage = vect_to_frame(get_mode(U, i))
#     axes.append(fig.add_subplot(3, 7, i + 1))
#     subplot_title = "Basis " + str(i)
#     axes[-1].set_title(subplot_title)
#     plt.imshow(bimage)
# plt.show()


# time_svded_video = []
# for i in range(vectorized_frame_video.shape[1]):
#     x = vectorized_frame_video[:, i].reshape(1024, 768) - background[:, i].reshape(
#         1024, 768
#     )
#     time_svded_video.append(x)

# time_svded_video = []
# for frame, background_frame in zip(vectorized_frame_video, background):
#     x = vect_to_frame(frame) - vect_to_frame(background_frame)
#     time_svded_video.append(x)

time_svded_video = [
    resample_matrix(vect_to_frame(frame) - vect_to_frame(background_frame))
    for frame, background_frame in zip(
        np.transpose(vectorized_frame_video),
        np.transpose(background),
    )
]

os.chdir(image_save_location)
np.save("time_svded_video.npy", time_svded_video)

# twice_svd_video = []
# for frame in time_svded_video:
#     bandpass = svd_and_filter(
#         frame, highest_kept_mode=0, lowest_kept_mode=400
#     )
#     fill = np.mean(frame)
#     bandpass = np.where(np.abs(bandpass) < 0.3, bandpass, fill)
#     twice_svd_video.append(bandpass)
# twice_svd_video

# alph = list(string.ascii_lowercase)

# aa = []
# for i in alph:
#     for j in alph:
#         aa.append(i + j)

os.chdir(image_save_location)
# for i in range(len(twice_svd_video)):
for i, frame in enumerate(time_svded_video[:10]):
    d1 = matrix_to_image(frame)
    im = Image.fromarray(d1.astype(np.uint8))
    # a = aa[i // 10]
    # n = str(i % 10)
    # imname = "pod" + a + n + ".bmp"
    # imname = "pod" + i + ".bmp"
    im.save("pod" + str(i) + ".bmp")
