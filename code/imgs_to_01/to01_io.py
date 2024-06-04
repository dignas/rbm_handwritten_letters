from skimage import io
import argparse
import os
import glob


def get_args():
	current_dir = os.getcwd()
	default_output_dir = 'out'

	parser = argparse.ArgumentParser(description="Flatten .png images in all subdirectories to 0-1 colors using spectral methods")
	parser.add_argument('-d', default=current_dir, required=False, type=str, help='input directory (default: %(default)s)')
	parser.add_argument('-o', default=os.path.join(current_dir, default_output_dir), required=False, type=str, help='output directory (default: %(default)s)')

	args = parser.parse_args()

	if not os.path.isdir(args.d):
		print("input directory does not exist")
		exit(1)

	return args.d, args.o


def list_files(d):
	return glob.glob(d + '/**/*.png', recursive=True)


def create_output_dir(o):
	if not os.path.isdir(o):
		os.makedirs(o, exist_ok=True)


def read_file(f):
	img = io.imread(f)
	img = img[:,:,:3]
	return img


def save_file(img, name):
	if not os.path.isdir(os.path.dirname(name)):
		os.makedirs(os.path.dirname(name), exist_ok=True)
	io.imsave(name, img, check_contrast=False)
