import numpy as np
import scipy.ndimage as nd


def augment_dataset(X, Y):
	shift_direction = [
		[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
		[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
		[[0, 0, 0], [0, 0, 1], [0, 0, 0]],
		[[0, 0, 0], [0, 0, 0], [0, 1, 0]],
	]

	X = np.concatenate([X] + [np.apply_along_axis(__shift, 1, X, d) for d in shift_direction])
	Y = np.concatenate([Y for _ in range(5)], axis=0)

	return X, Y

def __shift(x, w):
	return nd.convolve(np.reshape(x, (32, 32)), weights=w, mode="constant").ravel()
