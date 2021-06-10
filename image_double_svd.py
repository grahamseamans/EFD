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
from tensorflow.python.ops.gen_linalg_ops import svd
from tensorflow.python.ops.math_ops import reduce_mean
from sklearn.linear_model import OrthogonalMatchingPursuit
import pydmd

# resample_factor = 5

# read in data
matlab_save_location = os.path.join(os.getcwd(), "data", "tir", "july9_2200_first")
image_save_location = os.path.join(os.getcwd(), "data", "images")
video_matrix = np.load(image_save_location + "/video_matrix_normalized.npy")


def to_tf_float32(Tensors):
    return [tf.cast(tensor, tf.float32) for tensor in Tensors]


def to_tf_float64(Tensors):
    return [tf.cast(tensor, tf.float64) for tensor in Tensors]


def display(Tensors):
    a = ""
    for Tensor in Tensors:
        a += str(Tensor.dtype) + " "
    print(a)
    a = ""
    for Tensor in Tensors:
        print(Tensor.shape, end=" ")
    print()


def svd_decomp(tensor):
    S, U, V = tf.linalg.svd(tensor)
    return U, tf.linalg.diag(S), tf.transpose(V)
    # U, S, VT = tf.linalg.svd(data, full_matricies)
    # S = tf.diag.diag(S)
    # return U, tfS, VT


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


def svd_filter(U, S, VT, h, l):
    U = U[:, h:l]
    S = S[h:l, h:l]
    VT = VT[h:l, :]
    return U @ S @ VT


def svd_and_filter(tensor, h, l):
    U, S, VT = svd_decomp(tensor)
    return svd_filter(U, S, VT, h=h, l=l)
    # display([U, S, VT])


def get_mode(u, basis):
    return u[:, basis]


def vect_to_frame(vect):
    return vect.reshape(1024, 768)


def resample_matrix(matrix):
    return matrix[::resample_factor, ::resample_factor]


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


"""
video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float64)
# video_matrix = tf.keras.layers.LayerNormalization(axis=1)(video_matrix)
# video_matrix = tf.keras.layers.LayerNormalization(axis=1)(video_matrix).numpy()
display([video_matrix])
video_tensor = tf.reshape(video_matrix, (1024, 768, 400))
video_matrix = video_matrix.numpy()
print(0)
random_matrix = np.random.permutation(video_matrix.shape[0] * video_matrix.shape[1])
random_matrix = random_matrix.reshape(video_matrix.shape[1], video_matrix.shape[0])
compression_matrix = random_matrix / np.linalg.norm(random_matrix)

dmd = pydmd.CDMD(svd_rank=2, compression_matrix=compression_matrix)
dmd.fit(video_matrix)

background = dmd.reconstructed_data.real
print(0)

background = tf.convert_to_tensor(background, dtype=tf.float64)
display([background])
video_background = tf.reshape(background, (1024, 768, 400))

display([video_background, video_tensor])

deback = tf.subtract(video_tensor, video_background)

display([background])
# video_tensor_tf = np.reshape(video_matrix, (1024, 768, 399))
# display([video_background, video_tensor_tf])
# time_svd_video = video_tensor_tf - video_background

# num_basis_vectors = 20
# axes = []
# fig = plt.figure()
# for i in range(0, num_basis_vectors):
#     bimage = np.transpose(deback[:, :, i + 300])
#     axes.append(fig.add_subplot(4, 5, i + 1))
#     subplot_title = "Frame " + str(i)
#     axes[-1].set_title(subplot_title)
#     plt.imshow(bimage)
# plt.show()

print(1)
twice_svd_video = []
for frame in range(np.shape(deback)[2]):
    bandpass = svd_and_filter(deback[:, :, frame], h=0, l=400)
    twice_svd_video.append(bandpass)
twice_svd_video
print(2)

num_basis_vectors = 20
axes = []
fig = plt.figure()
for i in range(0, num_basis_vectors):
    print(i)
    bimage = tf.transpose(twice_svd_video[i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)
plt.show()

assert 2 == 3
"""

video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float64)
first_frame = video_matrix[:, 0]


def get_left_right_snapshot(matrix):
    return (matrix[:, :-1], matrix[:, 1:])


X, X_prime = get_left_right_snapshot(video_matrix)


def get_subsample_indicies(num_pixels, min):
    return np.random.choice(
        num_pixels, min if num_pixels > min else num_pixels, replace=False
    )


# UNTESTED
# def subsample_pixesl(desired_pixels, video_matrix):
#     actual_pixels, frames = video_matrix.shape
#     return tf.gather(
#         video_matrix,
#         np.random.choice(
#             actual_pixels,
#             actual_pixels if desired_pixels > actual_pixels else desired_pixels,
#             replace=False,
#         )
#     )

"""
okay so the video is nxm
n = pixesl
m = frames or time

compression matrix is px(n or m)
so it's some parameter, by n (number of pixels..)
I assume that you want p to be less than m, or else you get something taller out...
but it must be >= 1 according to the paper
"""

# random_matrix = np.random.permutation(video_matrix.shape[0] * video_matrix.shape[1])
# random_matrix = random_matrix.reshape(video_matrix.shape[1], video_matrix.shape[0])
# compression_matrix = random_matrix / np.linalg.norm(random_matrix)

# chosen_pixels = get_subsample_indicies(pixels, 5000)
# compressed = tf.gather(video_matrix, chosen_pixels)


pixels, frames = video_matrix.shape

C = tf.random.normal(shape=[1000, pixels], dtype=tf.float64)
compressed = C @ video_matrix
display([compressed, C, video_matrix])

first_frame_compressed = compressed[:, 0]
Y, Y_prime = get_left_right_snapshot(compressed)
del compressed
U, S, VT = svd_decomp(Y)
V = tf.transpose(VT)
display([U, S, V])
del Y
S_inv = tf.linalg.pinv(S)
del S
display([tf.transpose(U, conjugate=True), Y_prime, V, S_inv])
A_hat = tf.transpose(U, conjugate=True) @ Y_prime @ V @ S_inv
del U
eigen_values, W = tf.linalg.eig(A_hat)
W = tf.cast(W, dtype=tf.float64)
eigen_values = tf.cast(eigen_values, dtype=tf.float64)
del A_hat
Phi = X_prime @ V @ S_inv @ W
del Y_prime
del W
del VT
del X_prime
del S_inv
display([Phi, first_frame])
# b = OrthogonalMatchingPursuit(n_nonzero_coefs=200).fit(Phi, first_frame).coef_
# b = tf.convert_to_tensor(b, dtype=tf.float64)
# b = tf.cast(
#     tf.linalg.lstsq(tf.cast(Phi, tf.float64), tf.cast(first_frame, tf.float64)),
#     tf.float32,
# )
b = tf.linalg.lstsq(Phi, tf.expand_dims(first_frame, -1))
b = b[:, 0]
B = tf.linalg.diag(b)
Vander = tf.convert_to_tensor(np.vander((eigen_values)), dtype=tf.float64)

print(b)
print(Phi)
print(Vander)
video_matrix = video_matrix[:, :-1]

# U, S, VT = svd_decomp(video_matrix)
background = svd_filter(Phi, B, Vander, h=0, l=5)
# background = svd_filter(U, S, VT, h=0, l=5)

video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 399))
video_background = tf.reshape(background, (1024, 768, 399))
# video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 400))
# video_background = tf.reshape(background, (1024, 768, 400))

time_svd_video = video_tensor_tf - video_background

num_basis_vectors = 20
axes = []
fig = plt.figure()
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_background[:, :, i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)
# plt.show()

num_basis_vectors = 20
axes = []
fig = plt.figure()
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_tensor_tf[:, :, i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(time_svd_video[:, :, i])
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
