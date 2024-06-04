import numpy as np


class RBM_Sampler:

	__K = 1000
	__seed = 12345


	def __init__(self, seed=None):
		if seed is not None:
			self.__seed = seed

		np.random.seed(self.__seed)

	def rbm_sample(self, RBM, K=None):
		W = RBM.components_
		b = RBM.intercept_visible_
		c = RBM.intercept_hidden_
		v0 = np.random.binomial(1, 0.5, b.shape)
		if K is None:
			K = self.__K
		return self.__sample_k_steps(v0, K, W, b, c)

	def __sigma(self, x):
		return 1/(1 + np.exp(-x))

	def __gibbs_sampling_step(self, v, W, b, c):
		ph = self.__sigma(W @ v + c)
		h = np.random.binomial(1, ph, ph.shape)
		pv = self.__sigma(W.T @ h + b)
		return np.random.binomial(1, pv, pv.shape)

	def __sample_k_steps(self, v0, k, W, b, c):
		v = v0
		for i in range(k):
			v = self.__gibbs_sampling_step(v, W, b, c)
		return v
