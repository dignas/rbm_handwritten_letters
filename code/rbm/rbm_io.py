from domain import cyryllic

import pickle
import os
import argparse
from enum import Enum
import glob
import numpy as np
from skimage import io, color
import datetime


class RBM_IO:

	dataset_dir = None
	output_dir = None
	dataset = None

	rbm_path = None
	no_samples = None
	sampling_steps = None
	sampling_output = None


	def __init__(self, load_sample=False):
		if load_sample:
			self.__get_args_load_sample()
		else:
			self.__get_args_train()

	def save_model(self, model, name):
		fullname = os.path.join(self.output_dir, name)
		if not os.path.isdir(os.path.dirname(fullname)):
			os.makedirs(os.path.dirname(fullname), exist_ok=True)

		f = open(fullname, 'wb')
		pickle.dump(model, f)
		f.close()

	def load_dataset(self):
		if self.dataset == Datasets.CYRYLLIC.value:
			return self.__load_cyryllic(self.dataset_dir)
		
	def load_rbm(self):
		f = open(self.rbm_path, 'rb')
		rbm = pickle.load(f)
		f.close()

		return rbm
	
	def save_samples(self, samples, shape):
		fullpath = os.path.join(self.sampling_output, str(datetime.datetime.now().timestamp()).replace('.', '_'))
		if not os.path.isdir(fullpath):
			os.makedirs(fullpath, exist_ok=True)

		for i, s in enumerate(samples):
			s_r = np.reshape(s, shape)
			img = color.label2rgb(s_r, kind='overlay', bg_label=0, bg_color='white', colors=['black'])
			img = (img * 255).astype(np.uint8)
			io.imsave(os.path.join(fullpath, f's{i}.png'), img, check_contrast=False)

	def __load_cyryllic(self, dirname):
		fs = glob.glob(dirname + '/**/*.png', recursive=True)

		X = np.array([
			[1 if px[0] == 0 else 0 for px in np.reshape(io.imread(f), (-1, 3))]
			for f in fs
		])

		y = np.array([
			cyryllic.label_by_fname(f)
			for f in fs
		])

		return X, y
	
	def __get_args_train(self):
		current_dir = os.getcwd()
		default_output = 'save_rbm'
		default_dataset = Datasets.CYRYLLIC.value

		parser = argparse.ArgumentParser(description="Train RBM to generate and recognize images and save the trained model")
		parser.add_argument('-d', default=current_dir, required=False, type=str, help='dataset directory (default: %(default)s)')
		parser.add_argument('-o', default=os.path.join(current_dir, default_output), required=False, type=str, help='output file for the trained model (default: %(default)s)')
		parser.add_argument('-D', default=default_dataset, required=False, type=str, help='dataset name (default: %(default)s)')
		

		args = parser.parse_args()

		if not os.path.isdir(args.d):
			print("input directory does not exist")
			exit(1)

		self.dataset_dir = args.d
		self.output_dir = args.o
		self.dataset = args.D

	def __get_args_load_sample(self):
		default_rbm_path = None
		default_no_samples = 50
		default_sampling_steps = 1000
		default_sampling_output = 'save_samples'

		parser = argparse.ArgumentParser(description="Train RBM to generate and recognize images and save the trained model")
		parser.add_argument('-d', default=default_rbm_path, required=False, type=str, help='path to saved RBM')
		parser.add_argument('-n', default=default_no_samples, required=False, type=int, help='number of samples (default: %(default)d)')
		parser.add_argument('-k', default=default_sampling_steps, required=False, type=int, help='number of Gibbs sampling steps per sample (default: %(default)d)')
		parser.add_argument('-o', default=default_sampling_output, required=False, type=str, help='sampling output directory (default: %(default)s)')

		args = parser.parse_args()

		if not os.path.isfile(args.d):
			print("input file does not exist")
			exit(1)

		self.rbm_path = args.d
		self.no_samples = args.n
		self.sampling_steps = args.k
		self.sampling_output = args.o


class Datasets(Enum):
	CYRYLLIC = 'cyryllic'
