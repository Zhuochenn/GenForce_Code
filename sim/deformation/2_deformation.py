import os
import configargparse
import numpy as np
from tqdm.auto import tqdm

def get_parser():

	parser = configargparse.ArgumentParser(description="Deformation Field Generation for Pose Estimation",
										   config_file_parser_class=configargparse.YAMLConfigFileParser,
										   formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument("--config", is_config_file=True, default="sim/path.yml")
	parser.add_argument("--indenters", default="sim/assets/indenters/input/npy_100000")
	parser.add_argument("--dir_output", default="sim/assets/indenters/output/npz")
	parser.add_argument("--stepX", type=int, default=4)
	parser.add_argument("--stepY", type=int, default=4)
	parser.add_argument("--stepZ", type=float, default=0.3)
	parser.add_argument("--maxZ",  type=float, default=1.5)

	return parser

if __name__ == "__main__":

	parser = get_parser()
	args = parser.parse_args()
	stepX, stepY, stepZ, maxZ = args.stepX, args.stepY, args.stepZ, args.maxZ
	Indenters = os.listdir(args.indenters)
	for obj in tqdm(Indenters):
		print(f"Collecting: {obj}")
		coor_x, coor_y = np.arange(-stepX, stepX+1, stepX), np.arange(-stepY, stepY+1, stepY)
		xv, yv = np.meshgrid(coor_x,coor_y)
		xv ,yv = xv.flatten(), yv.flatten()
		for i in tqdm(range(xv.shape[0])):
			for z in np.arange(stepZ,maxZ+0.1,stepZ):
				cmd = "python sim/deformation/gel_press.py --object " + obj[:-4] + " --dir_output " + args.dir_output + \
					  " --x " + str(xv[i]) + " --y " +  str(yv[i])+ " --depth " +  str(z)
				os.system(cmd)
