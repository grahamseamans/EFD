from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

matlab_save_location = os.path.join(os.getcwd(), "data", "tir", "july9_2200_first")
image_save_location = os.path.join(os.getcwd(), "data", "images")
video_matrix = np.load(image_save_location + "/video_matrix.npy")
video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float64)
first_frame = video_matrix[:, 0]


# X, X_prime = get_left_right_snapshot(video_matrix)
X = video_matrix
pixels, frames = video_matrix.shape

C = tf.random.normal(shape=[20, pixels], dtype=tf.float64)
display([C, video_matrix])
Y = C @ video_matrix
display([Y, tf.transpose(Y)])
B = Y @ tf.transpose(Y)
display([B, tf.transpose(B)])
B = 1 / 2 * (B + tf.transpose(B))
display([B])
D, T = tf.linalg.eig(B)
D = tf.cast(D, dtype=tf.float64)
T = tf.cast(T, dtype=tf.float64)
display([D])
S_s = tf.linalg.diag(tf.math.sqrt(D))
display([tf.transpose(Y), T, tf.linalg.pinv(S_s)])
V_s = tf.transpose(Y) @ T @ tf.linalg.pinv(S_s)
display([X, V_s])
U_s = X @ V_s
display([U_s])
U, S, QT = svd_decomp(U_s)
display([QT])
V = V_s @ tf.transpose(QT)
display([U, S, V])

background = svd_filter(U, S, tf.transpose(V), h=0, l=2)
video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 400))
video_background = tf.reshape(background, (1024, 768, 400))
time_svd_video = video_tensor_tf - video_background

num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("Modes")
for i in range(0, num_basis_vectors):
    bimage = tf.reshape(get_mode(U, i), [1024, 768])
    axes.append(fig.add_subplot(3, 7, i + 1))
    subplot_title = "Basis " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("video background")
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
plt.title("plain video")
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(video_tensor_tf[:, :, i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)

num_basis_vectors = 20
axes = []
fig = plt.figure()
plt.title("snapshotted")
for i in range(0, num_basis_vectors):
    bimage = tf.transpose(time_svd_video[:, :, i])
    axes.append(fig.add_subplot(4, 5, i + 1))
    subplot_title = "Frame " + str(i)
    axes[-1].set_title(subplot_title)
    plt.imshow(bimage)
plt.show()
