# Graham Seamans
"""

"""

from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import cProfile
import pstats
import os

video_matrix = np.load(os.path.join(npy_path, "video_matrix_tf.npy"))
video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float64)


profiler = cProfile.Profile()
profiler.enable()

# U, S, VT = csvd_1(video_matrix, p=40)
U, S, VT = csvd_2(video_matrix, p=40)
# U, S, VT = svd_decomp(video_matrix)
# U, S, VT = cdmd(video_matrix, p=500)

profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs().sort_stats("tottime").print_stats(10)

background = svd_filter(U, S, VT, h=0, l=2)

# For test the SVD's
video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 400))
video_background = tf.reshape(background, (1024, 768, 400))

# For testing the cDMD
# video_matrix = video_matrix[:, 1:]
# video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 399))
# video_background = tf.reshape(background, (1024, 768, 399))

time_svd_video = video_tensor_tf - video_background

Tensor_to_video(time_svd_video[:, :, :200], os.path.join(video_path, "snapshot"))

assert 2 == 3
num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("Modes")
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(tf.reshape(get_mode(U, i), [1024, 768]))
    axes.append(fig.add_subplot(4, 5, i + 1))
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("video background")
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_background[:, :, i + 300])
    axes.append(fig.add_subplot(4, 5, i + 1))
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("plain video")
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_tensor_tf[:, :, i + 300])
    axes.append(fig.add_subplot(4, 5, i + 1))
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("snapshotted")
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(time_svd_video[:, :, i + 300])
    axes.append(fig.add_subplot(4, 5, i + 1))
    plt.imshow(bimage)
plt.show()

"""
U, S, VT = csvd_1(video_matrix, p=40)
background_1 = svd_filter(U, S, VT, h=0, l=2)
U, S, VT = csvd_2(video_matrix, p=40)
background_2 = svd_filter(U, S, VT, h=0, l=2)
U, S, VT = svd_decomp(video_matrix)
background = svd_filter(U, S, VT, h=0, l=2)

one_vs_two = background_1 - background_2
one_vs_plain = background_1 - background
two_vs_plain = background_2 - background


def get_stats(Tensor):
    std = tf.math.reduce_std(Tensor)
    mean = tf.math.reduce_mean(Tensor)
    return std, mean


print("one_vs_two", get_stats(one_vs_two))
print("one_vs_plain", get_stats(one_vs_plain))
print("two_vs_plain", get_stats(two_vs_plain))
"""
