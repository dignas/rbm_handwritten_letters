from rbm_io import RBM_IO
from rbm_sampler import RBM_Sampler

import numpy as np


if __name__ == "__main__":
	io = RBM_IO(load_sample=True)
	sampler = RBM_Sampler()

	rbm = io.load_rbm()
	
	samples = np.array([
		sampler.rbm_sample(rbm, io.sampling_steps)
		for i in range(io.no_samples)
	])

	io.save_samples(samples, (32, 32))
