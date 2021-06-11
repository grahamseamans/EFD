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

matlab_save_location = os.path.join(os.getcwd(), "data", "tir", "july9_2200_first")
image_save_location = os.path.join(os.getcwd(), "data", "images")
video_save_location = os.path.join(os.getcwd(), "data", "videos")
# video_matrix = np.load(image_save_location + "/video_matrix.npy")
# video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float64)
# video_matrix = tf.keras.layers.LayerNormalization(axis=1)(video_matrix)
# video_tensor = tf.reshape(video_matrix, (1024, 768, 400))
# np.save(image_save_location + "/video_matrix_normalized_64.npy", video_matrix.numpy())

# def tensor_to_video(Tensor):
#     display([Tensor])
#     height, width, num_frames = Tensor.shape
#     print(height, width, num_frames)
#     filename = os.path.join(video_save_location, "test_video.avi")
#     codec_id = "mp4v"  # ID for a video codec.
#     fourcc = cv2.VideoWriter_fourcc(*code)
#     out = cv2.VideoWriter(
#         "test_video.avi", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width, height)
#     )
#     for i in range(num_frames):
#         out.write(Tensor[:, :, i])


def csvd_1(video_matrix, p):
    X = video_matrix
    pixels, frames = video_matrix.shape
    C = tf.random.normal(shape=[p, pixels], dtype=tf.float64)
    Y = C @ video_matrix
    B = Y @ tf.transpose(Y)
    B = 1 / 2 * (B + tf.transpose(B))
    D, T = tf.linalg.eig(B)
    D = tf.cast(D, dtype=tf.float64)
    T = tf.cast(T, dtype=tf.float64)
    S_s = tf.linalg.diag(tf.math.sqrt(D))
    V_s = tf.transpose(Y) @ T @ tf.linalg.pinv(S_s)
    U_s = X @ V_s
    U, S, QT = svd_decomp(U_s)
    V = V_s @ tf.transpose(QT)
    return U, S, tf.transpose(V)


def csvd_2(video_matrix, p):
    X = video_matrix
    pixels, frames = video_matrix.shape
    C = tf.random.normal(shape=[p, pixels], dtype=tf.float64)
    Y = C @ video_matrix
    T, S_s, VT_s = svd_decomp(Y)
    V_s = tf.transpose(VT_s)
    U, S, Q = svd_decomp(X @ V_s)
    V = V_s @ tf.transpose(Q)
    return U, S, tf.transpose(V)


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


def pod(data):
    return svd_decomp(np.cov(data))


def matrix_to_image(a):
    return (255 * (a - np.min(a)) / np.ptp(a)).astype(int)


def svd_filter(U, S, VT, h, l):
    U = U[:, h:l]
    S = S[h:l, h:l]
    VT = VT[h:l, :]
    return U @ S @ VT


def svd_and_filter(tensor, h, l):
    U, S, VT = svd_decomp(tensor)
    return svd_filter(U, S, VT, h=h, l=l)


def get_mode(u, basis):
    return u[:, basis]


def vect_to_frame(vect):
    return vect.reshape(1024, 768)


def resample_matrix(matrix):
    return matrix[::resample_factor, ::resample_factor]


def get_left_right_snapshot(matrix):
    return (matrix[:, :-1], matrix[:, 1:])


def get_subsample_indicies(num_pixels, min):
    return np.random.choice(
        num_pixels, min if num_pixels > min else num_pixels, replace=False
    )


# UNTESTED
def subsample_pixesl(desired_pixels, video_matrix):
    actual_pixels, frames = video_matrix.shape
    return tf.gather(
        video_matrix,
        np.random.choice(
            actual_pixels,
            actual_pixels if desired_pixels > actual_pixels else desired_pixels,
            replace=False,
        ),
    )
