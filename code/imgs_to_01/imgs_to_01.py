import to01_io as myio
from to01_alg import to01_alg

import os
import numpy as np


if __name__ == '__main__':
	d, o = myio.get_args()
	myio.create_output_dir(o)

	fs = myio.list_files(d)

	count = 0

	for f in fs:
		img = myio.read_file(f)
		end_img = to01_alg(img)

		relp = os.path.relpath(f, d)
		myio.save_file(end_img, os.path.join(o, relp))

		if count % 100 == 0:
			print(f"converted {count} imgs...")
		count += 1
