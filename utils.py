from re import A
import numpy as np
from numpy.lib.function_base import disp
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2
from tensorflow.python.ops.math_ops import reduce_min

npy_path = os.path.join(os.getcwd(), "data", "npy")
video_path = os.path.join(os.getcwd(), "data", "videos")


def Tensor_to_video(Tensor, path):
    # try l2 norm and then min_max
    Tensor = tf.keras.utils.normalize(Tensor, axis=2, order=13)
    video = tf.cast(255 * min_max_norm(Tensor), tf.uint8)
    width, height, frames = video.shape
    _fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(path + ".mp4", _fourcc, 5, (width, height))
    for i in range(frames):
        img = video[:, :, i]
        img = tf.repeat(tf.expand_dims(tf.transpose(img), axis=-1), 3, axis=2).numpy()
        out.write(img)


def min_max_norm(Tensor):
    min = tf.reduce_min(Tensor)
    max = tf.reduce_max(Tensor)
    return (Tensor - min) / (max - min)


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


def csvd_2(matrix, p):
    X = matrix
    pixels, frames = X.shape
    C = tf.random.normal(shape=[p, pixels], dtype=tf.float64)
    Y = C @ X
    T, S_s, VT_s = svd_decomp(Y)
    V_s = tf.transpose(VT_s)
    U, S, Q = svd_decomp(X @ V_s)
    V = V_s @ tf.transpose(Q)
    return U, S, tf.transpose(V)


def cdmd(video_matrix, p):
    first_frame = video_matrix[:, 0]
    X, X_prime = get_left_right_snapshot(video_matrix)
    pixels, frames = video_matrix.shape
    C = tf.random.normal(shape=[p, pixels], dtype=tf.float64)
    compressed = C @ video_matrix
    Y, Y_prime = get_left_right_snapshot(compressed)
    U, S, VT = svd_decomp(Y)
    V = tf.transpose(VT)
    S_inv = tf.linalg.pinv(S)
    A_hat = tf.transpose(U, conjugate=True) @ Y_prime @ V @ S_inv
    eigen_values, W = tf.linalg.eig(A_hat)
    W = tf.cast(W, dtype=tf.float64)
    eigen_values = tf.cast(eigen_values, dtype=tf.float64)
    Phi = X_prime @ V @ S_inv @ W
    b = tf.linalg.lstsq(Phi, tf.expand_dims(first_frame, -1))
    b = b[:, 0]
    B = tf.linalg.diag(b)
    Vander = tf.convert_to_tensor(np.vander((eigen_values)), dtype=tf.float64)
    return Phi, B, Vander


def frame_svd(Tensor, p):
    _, _, num_frames = Tensor.shape
    # video = tf.transpose(Tensor, [2, 0, 1])
    # video = tf.vectorized_map(svd_filter_wrapper, video, fallback_to_while_loop=True)
    # display([video])
    # assert 2 == 3
    frames = []
    for i in range(num_frames):
        frame = Tensor[:, :, i]
        frame = svd_and_filter(frame, 0, 200, p)
        frames.append(frame)
    x = tf.stack(frames, axis=-1)
    display([x])
    return x


def svd_filter_wrapper(frame):
    return svd_and_filter(frame, 0, 200, 400)


def psd(VT, S, num_basis):
    sl = tf.linalg.diag_part(S)
    rwl = []
    rfl = []
    for i in range(num_basis):
        rp = VT[i, :] * sl[i]
        f, P = signal.welch(rp)
        rwl.append(P)
        rfl.append(f)

    plt.subplots()
    leg = []
    con = 0
    for f, p in zip(rfl, rwl):
        print(f, p)
        plt.loglog(f, p)
        leg.append("Mode " + str(con + 1) + " S value " + str(int(sl[con])))
        con += 1

    plt.legend(leg)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [T**2/Hz]")

    slope = -5 / 3
    dx = 0.1
    x0 = 0.004
    # y0 = 10
    # y = np.array(rwl)

    # y0 = y[:, 0].mean()
    # display([y, y0])

    # x_input = np.logspace(0.0, 0.5, 127)
    # x_axis = np.linspace(0.0, 0.5, 127)
    # y = [10000 * (x ** slope) + y0 for x in x_input]
    # print(y)
    # plt.loglog(x_axis, y)

    # print(x_axis, y)
    # print(y)

    x1, y1 = [x0, x0 + dx], [x0 ** slope / 2000, (x0 + dx) ** slope / 2000]
    print(x1, y1)
    plt.plot(x1, y1)
    plt.text(0.02, 1, "-5/3 Slope", bbox=dict(facecolor="none", edgecolor="black"))
    plt.show()


def background_thresholding(background, video, tao):
    return tf.where((video - background) > tao, video, 0)


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


def svd_and_filter(tensor, h, l, p):
    U, S, VT = csvd_2(tensor, p)
    return svd_filter(U, S, VT, h=h, l=l)


def get_mode(u, basis):
    return u[:, basis]


def vect_to_frame(vect):
    return vect.reshape(1024, 768)


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
