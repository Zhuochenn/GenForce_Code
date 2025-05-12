import open3d as o3d
import numpy as np
import os
from tqdm.auto import tqdm

def npz2ply(file_in_path,file_out_path):
    coordinates = np.load(file_in_path)
    x,y,z = coordinates['p_xpos_list'][0],coordinates['p_ypos_list'][0],coordinates['p_zpos_list'][0]
    points = np.vstack((x,y,z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_out_path,point_cloud)

def npz2stl(file_in_path,file_out_path):

    coordinates = np.load(file_in_path)
    x,y,z = coordinates['p_xpos_list'][-1],coordinates['p_ypos_list'][-1],coordinates['p_zpos_list'][-1]
    points = np.vstack((x,y,z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(point_cloud.normals)  
    if np.any(np.isnan(normals)):  
        print("Warning: There are NaN values in the normals. Please check the point cloud quality.")  
        exit(1)  
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)  
    if len(mesh.triangles) == 0:  
        print("Warning: The mesh reconstruction resulted in no triangles. Check point cloud data and normals.")  
        exit(1)  

    # Validate mesh normals  
    mesh.compute_vertex_normals()  
    mesh.normalize_normals()  

    # Save the mesh to an .stl file  
    success = o3d.io.write_triangle_mesh(file_out_path, mesh)  

    if success:  
        print(f"Mesh saved to {file_out_path}")  
    else:  
        print("Failed to save mesh. Ensure all steps are completed correctly.")  


root = "sim/assets/indenters/output"
out_root_npz = "sim/assets/indenters/output/npz"
out_root_stl = "sim/assets/indenters/output/stl"
indenters  = os.listdir(out_root_npz)
if not os.path.exists(out_root_stl):
    os.makedirs(out_root_stl)

for indenter in tqdm(indenters):
    out_dir = os.path.join(out_root_stl,indenter)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    npz_f = os.listdir(os.path.join(out_root_npz,indenter))
    for fs in tqdm(npz_f):
        name = fs[0:-4]
        file_in_path = os.path.join(out_root_npz,indenter,fs)
        file_out_path = os.path.join(out_dir,name+".stl")
        npz2stl(file_in_path,file_out_path)
    print(f'{indenter} done!')

