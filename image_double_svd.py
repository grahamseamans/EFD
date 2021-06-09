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
from numpy import float32, linalg
from numpy.core.numeric import full
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.python.ops.math_ops import reduce_mean
from sklearn.linear_model import OrthogonalMatchingPursuit
import pydmd

resample_factor = 5


def display(Tensors):
    a = ""
    for Tensor in Tensors:
        a += str(Tensor.dtype) + " "
    print(a)
    a = ""
    for Tensor in Tensors:
        print(Tensor.shape, end=" ")
    print()


def svd_decomp(data, full_matricies=False):
    U, S, VT = np.linalg.svd(data, full_matricies)
    S = np.diag(S)
    return U, S, VT


def pod(data):
    return svd_decomp(np.cov(data))


def matrix_to_image(a):
    return (255 * (a - np.min(a)) / np.ptp(a)).astype(int)


# @jit(nopython=True)
# def svd_filter(U, S, VT, h, l):
#     U = np.ascontiguousarray(U[:, h:l])
#     S = np.ascontiguousarray(S[h:l, h:l])
#     VT = np.ascontiguousarray(VT[h:l, :])
#     return U @ S @ VT


def svd_and_filter(tensor, highest_kept_mode, lowest_kept_mode):
    U, S, VT = svd_decomp(tensor)
    return svd_filter(
        U, S, VT, highest_kept_mode=highest_kept_mode, lowest_kept_mode=lowest_kept_mode
    )


def get_mode(u, basis):
    return u[:, basis]


def vect_to_frame(vect):
    return vect.reshape(1024, 768)


def resample_matrix(matrix):
    return matrix[::resample_factor, ::resample_factor]


# read in data
matlab_save_location = os.path.join(os.getcwd(), "data", "tir", "july9_2200_first")
image_save_location = os.path.join(os.getcwd(), "data", "images")

"""
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


os.chdir(image_save_location)
np.save("vectorized_frame_video.npy", vectorized_frame_video)
"""

video_matrix = np.load(image_save_location + "/video_matrix.npy")

video_matrix = tf.convert_to_tensor(video_matrix)
video_matrix = tf.keras.layers.LayerNormalization(axis=1)(video_matrix)
# video_matrix = tf.keras.layers.LayerNormalization(axis=1)(video_matrix).numpy()

# dmd = pydmd.DMD(svd_rank=3, exact=False)
# dmd.fit(video_matrix)

# background = dmd.reconstructed_data.real
# display([background])

# background = tf.convert_to_tensor(background)
# display([background])
# video_background = tf.reshape(background, (400, 1024, 768))

# # video_tensor_tf = np.reshape(video_matrix, (1024, 768, 399))
# # display([video_background, video_tensor_tf])
# # time_svd_video = video_tensor_tf - video_background

# assert not tf.math.equal(video_background[0], video_background[1])
# num_basis_vectors = 20
# axes = []
# fig = plt.figure()
# for i in range(0, num_basis_vectors):
#     bimage = np.transpose(video_background[i])
#     axes.append(fig.add_subplot(4, 5, i + 1))
#     subplot_title = "Frame " + str(i)
#     axes[-1].set_title(subplot_title)
#     plt.imshow(bimage)
# plt.show()


video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float32)
first_frame = video_matrix[:, 0]


def get_left_right_snapshot(matrix):
    return (matrix[:, :-1], matrix[:, 1:])


X, X_prime = get_left_right_snapshot(video_matrix)


def get_subsample_indicies(num_pixels, min):
    return np.random.choice(
        num_pixels, min if num_pixels > min else num_pixels, replace=False
    )


pixels, frames = video_matrix.shape
chosen_pixels = get_subsample_indicies(pixels, 5000)
compressed = tf.gather(video_matrix, chosen_pixels)
# display([compressed])
# assert 2 == 3
first_frame_compressed = compressed[:, 0]

# del video_matrix
Y, Y_prime = get_left_right_snapshot(compressed)
del compressed
S, U, VT = tf.linalg.svd(Y, full_matrices=False)
S = tf.linalg.diag(S)
del Y
S_inv = tf.linalg.pinv(S)
del S
A_hat = tf.transpose(U, conjugate=True) @ Y_prime @ VT @ S_inv
del U
V, W = tf.linalg.eig(A_hat)
V = tf.cast(V, dtype=float32)
W = tf.cast(W, dtype=float32)
del A_hat
Phi = X_prime @ VT @ S_inv @ W
Phi_y = Y_prime @ VT @ S_inv @ W
del Y_prime
del W
del VT
del X_prime
del S_inv
# b = np.linalg.lstsq(Phi.numpy(), first_frame.numpy())
Phi_norm = tf.keras.layers.LayerNormalization(axis=1)(Phi)
first_frame_norm = tf.keras.layers.LayerNormalization(axis=0)(first_frame)
# Phi_norm = layer(Phi)
print(Phi)
print(Phi_norm)
b = OrthogonalMatchingPursuit().fit(Phi_norm, first_frame_norm).coef_
# b = OrthogonalMatchingPursuit().fit(Phi_y, first_frame_compressed).coef_
b = tf.convert_to_tensor(b, dtype=float32)
# b = tf.linalg.lstsq(Phi_y, first_frame_compressed)
B = tf.linalg.diag(b)
V = tf.convert_to_tensor(np.vander((V)), dtype=float32)


# video_tensor_np = []
# for frame in video_matrix.numpy().T:
#     print(frame.shape)
#     video_tensor_np.append(vect_to_frame(frame))
# video_tensor_np = np.stack(video_tensor_np, axis=1)
# print(video_matrix.shape, video_tensor_np.shape, video_tensor_tf.shape)
# print(video_tensor_tf - video_tensor_np)
# Phi = Phi.numpy()
# B = B.numpy()
# V = V.numpy()

# print(Phi.dtype, B.dtype, V.dtype)
# print(Phi.shape, B.shape, V.shape)


def svd_filter(U, S, VT, h, l):
    U = U[:, h:l]
    S = S[h:l, h:l]
    VT = VT[h:l, :]
    display([U, S, VT])
    return U @ S @ VT


# return U[:, h:l] @ S[h:l, h:l] @ VT[h:l, :]
print("phi", Phi)
print("B", B)
print("V", V)
video_matrix = video_matrix[:, :-1]
background = svd_filter(Phi, B, V, h=0, l=4)
# print(background)
video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 399))
video_background = tf.reshape(background, (1024, 768, 399))
display([video_background, video_tensor_tf])
# time_svd_video = video_tensor_tf - video_background

num_basis_vectors = 20
axes = []
fig = plt.figure()
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_background[:, :, i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_tensor_tf[:, :, i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)
plt.show()


# time_svded_video = []
# for frame, background_frame in zip(video_matrix, background)[20]:
#     x = vect_to_frame(frame) - vect_to_frame(background_frame)
#     time_svded_video.append(x)

# axes = []
# fig = plt.figure()
# for i, bimage in enumerate(time_svded_video[:20]):
#     axes.append(fig.add_subplot(3, 7, (i - 20) + 1))
#     subplot_title = "Basis " + str(i)
#     axes[-1].set_title(subplot_title)
#     plt.imshow(bimage)
# plt.show()


"""
U, S, VT = svd_decomp(video_matrix, full_matricies=False)

bandpass = high_low_svd_filter(U, S, VT, highest_kept_mode=0, lowest_kept_mode=400)
background = high_low_svd_filter(U, S, VT, highest_kept_mode=0, lowest_kept_mode=12)

# num_basis_vectors = 20
# axes = []
# fig = plt.figure()
# for i in range(0, num_basis_vectors):
#     bimage = vect_to_frame(get_mode(Phi, i))
#     axes.append(fig.add_subplot(3, 7, i + 1))
#     subplot_title = "Basis " + str(i)
#     axes[-1].set_title(subplot_title)
#     plt.imshow(bimage)
# plt.show()


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
        np.transpose(video_matrix),
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
"""
