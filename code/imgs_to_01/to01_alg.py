from skimage import color
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def to01_alg(img):
	labels_unassigned_colors = __kmeans(img)
	labels = __assign_color(labels_unassigned_colors)

	end_img = color.label2rgb(labels, kind='overlay', bg_label=0, bg_color='white', colors=['black'])

	return (end_img * 255).astype(np.uint8)


def __kmeans(img):
	kmeans = KMeans(n_clusters=2).fit(np.reshape(img, (-1, 3)))
	return np.reshape(kmeans.labels_, np.shape(img)[:-1])


def __assign_color(zero_one_labels):
	_, probe_size = np.shape(zero_one_labels)
	zero_one_labels_assigned = zero_one_labels
	if np.sum(zero_one_labels[0, :]) > probe_size / 2:
		zero_one_labels_assigned = 1 - zero_one_labels
	
	return zero_one_labels_assigned
