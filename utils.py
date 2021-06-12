# Graham Seamans
import numpy as np
import tensorflow as tf


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


def svd_and_filter(tensor, h, l):
    U, S, VT = svd_decomp(tensor)
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
