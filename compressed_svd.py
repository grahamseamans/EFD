from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

matlab_save_location = os.path.join(os.getcwd(), "data", "tir", "july9_2200_first")
image_save_location = os.path.join(os.getcwd(), "data", "images")
video_matrix = np.load(image_save_location + "/video_matrix.npy")
video_matrix = tf.convert_to_tensor(video_matrix, dtype=tf.float64)
first_frame = video_matrix[:, 0]

# U, S, VT = csvd_1(video_matrix, p=40)
U, S, VT = csvd_2(video_matrix, p=40)

background = svd_filter(U, S, VT, h=0, l=2)
video_tensor_tf = tf.reshape(video_matrix, (1024, 768, 400))
video_background = tf.reshape(background, (1024, 768, 400))
time_svd_video = video_tensor_tf - video_background

# tensor_to_video(video_tensor_tf)

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
