import open3d as o3d
import os
from tqdm.auto import tqdm
import argparse 
import numpy as np


def get_parser_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("--particle", type=int, default=100000)
	parser.add_argument("--dataset", default="sim/assets/indenters/input")
	args = parser.parse_args()

	return args

def stl2ply(url_stl,url_ply):

	YCBs  = os.listdir(url_stl)
	for obj in tqdm(YCBs):
		read_url = os.path.join(url_stl , obj)
		mesh = o3d.io.read_triangle_mesh(read_url) 
		write_url = os.path.join(url_ply , obj[:-3]+"ply")
		o3d.io.write_triangle_mesh(write_url,mesh)
	
	print("stl2ply done!")


def ply2pcd(url_ply,url_pcd, n_sample):
	
	YCBs  = os.listdir(url_ply)
	for obj in tqdm(YCBs):
		read_url = os.path.join(url_ply , obj)
		write_url = os.path.join(url_pcd , obj[:-3]+"pcd")
		mesh = o3d.io.read_triangle_mesh(read_url) 
		point_cloud = mesh.sample_points_uniformly(number_of_points=n_sample)  
		# o3d.visualization.draw_geometries([point_cloud], window_name="Dense Point Cloud") 
		o3d.io.write_point_cloud(write_url, point_cloud)   
	print("ply2pcd done!")


def pcd2npy(url_pcd,url_npy):

	YCBs  = os.listdir(url_pcd)
	for obj in tqdm(YCBs):
		npy_file = np.empty([0,3])
		read_url = os.path.join(url_pcd , obj)
		pcd = o3d.io.read_point_cloud(read_url)
		points = np.asarray(pcd.points)
		for i in range(np.shape(points)[0]):
			tmp = np.reshape(points[i,:],(1,3))
			npy_file = np.concatenate((npy_file,tmp))
		write_url = os.path.join(url_npy,obj[:-3]+"npy")
		np.save(write_url,npy_file.astype(np.float32))
		print(np.shape(npy_file))
		print("max: %f, %f, %f. min: %f, %f, %f."%(np.max(npy_file[:,0]),np.max(npy_file[:,1]),
			  np.max(npy_file[:,2]),np.min(npy_file[:,0]),np.min(npy_file[:,1]),
			  np.min(npy_file[:,2])))
		
	print("pcd2npy done!")

if __name__ == "__main__":

	args = get_parser_args()
	url_stl = os.path.join(args.dataset,"stl")
	url_ply = os.path.join(args.dataset,"ply")
	url_pcd = os.path.join(args.dataset,"pcd_"+str(args.particle))
	url_npy = os.path.join(args.dataset,"npy_"+str(args.particle))

	if not os.path.exists(url_ply):
		os.makedirs(url_ply)
	
	if not os.path.exists(url_pcd):
		os.makedirs(url_pcd)
	
	if not os.path.exists(url_npy):
		os.makedirs(url_npy)

	stl2ply(url_stl,url_ply)
	ply2pcd(url_ply,url_pcd, n_sample=args.particle)
	pcd2npy(url_pcd,url_npy)
