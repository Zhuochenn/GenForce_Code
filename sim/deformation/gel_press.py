import taichi as ti
import numpy as np
import argparse
import os
import sys
import argparse
from pytransform3d import rotations
import yaml

def get_args():  
    parser = argparse.ArgumentParser()  

    parser.add_argument("--config", default="sim/parameters.yml")  
    parser.add_argument("--particle", default="100000")  
    parser.add_argument("--dir_output", default="sim/assets/indenters/output/npz")  
    parser.add_argument("--dataset", default="sim/assets/indenters/input")  
    parser.add_argument("--object", default="wave")    
    parser.add_argument("--x", type=int, default=0)    
    parser.add_argument("--y", type=int, default=0)    
    parser.add_argument("--depth", type=float, default=1.5)    

    args = parser.parse_args()  

    return args  

def read_yaml_config(file_path):  
    try:  
        with open(file_path, 'r') as file:  
            data = yaml.safe_load(file)  
            return data  
    except FileNotFoundError:  
        print(f"Error: The file {file_path} was not found.")  
    except yaml.YAMLError as exc:  
        print(f"Error parsing YAML file: {exc}")  

    
args = get_args()
config = read_yaml_config(args.config)

if not os.path.exists(args.dir_output + "/" + args.object):
  os.makedirs(args.dir_output + "/" + args.object)
  

# taichi initialization
ti.init(arch=ti.gpu) 

# world initialization
t_ti = ti.field(dtype=ti.f32, shape=())
t_ti[None] = 0
dt = config["world"]["dt"]
x_offset = config["world"]["coordin_offset"]["x_offset"]
y_offset = config["world"]["coordin_offset"]["y_offset"]
z_offset = config["world"]["coordin_offset"]["z_offset"]

# elstomer
num_l, num_w = config["elastomer"]["particle"]["num_l"], config["elastomer"]["particle"]["num_w"]
num_h = config["elastomer"]["particle"]["num_h"]
l = config["elastomer"]["size"]["l"]
w = config["elastomer"]["size"]["w"]
h = config["elastomer"]["size"]["h"]

# indenter


pose_x, pose_y, pose_z = config["indenter"]["pose"]["x"], config["indenter"]["pose"]["y"], config["indenter"]["pose"]["z"]
pose_R, pose_P, pose_Y = config["indenter"]["pose"]["R"], config["indenter"]["pose"]["P"], config["indenter"]["pose"]["Y"]
indenting_depth = round(args.depth,1)
suffix = f"{args.x}_{args.y}_{indenting_depth}"
obj_name = os.path.join(args.dataset,"npy_" + args.particle, args.object + ".npy")
data = np.load(obj_name)
rotation_m = rotations.matrix_from_euler((pose_R, pose_P, pose_Y),0,1,2,True) # use extrinsic
data = np.matmul(rotation_m,data.T).T.astype(np.float32)+np.array([[x_offset+pose_x+args.x,y_offset+pose_y+args.y,pose_z]],dtype=np.float32)
# data = data+np.array([[x_offset+pose_x,y_offset+pose_y,pose_z]],dtype=np.float32)


# exit(0)
# grid
n_grid = config["grid"]["n_grid"]
l_grid = config["grid"]["length_grid"]
dx = l_grid / n_grid
inv_dx = 1 / dx
grid_v = ti.Vector.field(n=3, dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node mass

# particle initialization
particle_dis = l/(num_l-1) # particle distance
n_particles = num_l*num_w*num_h+np.shape(data)[0]
x = ti.Vector.field(3, dtype=float, shape=n_particles) # position
x_2d = ti.Vector.field(2, dtype=float, shape=n_particles) # 2d positions - this is necessary for circle visualization
v = ti.Vector.field(3, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id

# material
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = config["elastomer"]["modulus"], config["elastomer"]["poisson_ratio"]
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters - may change these later to model other materials

# np particle position 
p_xpos_list = np.empty([0,num_w * num_l])
p_ypos_list = np.empty([0,num_w * num_l])
p_zpos_list = np.empty([0,num_w * num_l])


@ti.kernel  
def substep():  
    # Clear grid  
    # print('start')
    for i, j, k in grid_m:  
        grid_v[i, j, k] = [0, 0, 0]  
        grid_m[i, j, k] = 0  
    # print('finish first step.')
    # Particle to Grid  
    for p in x:  
        if p >= n_particles:  # Ensure p is within bounds  
            continue  

        base = (x[p] * inv_dx - 0.5).cast(int)  
        fx = x[p] * inv_dx - base.cast(float)  
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  
        
        # Material properties  
        mu, la = mu_0, lambda_0  
        U, sig, V = ti.svd(F[p])  
        J = 1.0  
        for d in ti.static(range(3)):  
            J *= sig[d, d]  
        
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)  
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress  
        affine = stress + p_mass * C[p]  

        # Perform P2G with caching  
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  
            offset = ti.Vector([i, j, k])  
            dpos = (offset.cast(float) - fx) * dx  
            weight = w[i][0] * w[j][1] * w[k][2]  
            pose = base + offset
            if is_valid_vec(pose):
                grid_m[base + offset] += weight * p_mass  # mass transfer  
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos) 

    # print('finish second step.')
    # Normalize grid velocities  
    for i, j, k in grid_m:  
        if grid_m[i, j, k] > 0 and is_valid(i,j,k):  
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]  
            # Wall collision handling remains the same  
            if i < 3 and grid_v[i, j, k][0] < 0:          grid_v[i, j, k][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:          grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:          grid_v[i, j, k][2] = 0
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0
    # print('finish third step.')
    # Grid to Particle update  
    for p in x:  
        if p >= n_particles:  # Check for valid particle index  
            continue  

        base = (x[p] * inv_dx - 0.5).cast(int)  
        fx = x[p] * inv_dx - base.cast(float)  
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]  

        cached_v = ti.Vector.zero(float, 3)  
        cached_C = ti.Matrix.zero(float, 3, 3)  

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  
            pose = base + ti.Vector([i, j, k])
            if is_valid_vec(pose):
                dpos = ti.Vector([i, j, k]).cast(float) - fx  
                g_v = grid_v[pose]  
                weight = w[i][0] * w[j][1] * w[k][2]  

                # Accumulate cached values  
                cached_v += weight * g_v  
                cached_C += 4 * inv_dx * weight * g_v.outer_product(dpos)  

        # Update particle velocity and deformation gradient  
        v[p], C[p] = cached_v, cached_C  

        # Move particles safely  
        if p < num_l * num_w * 3:  
            v[p] = ti.Vector([0, 0, 0])  
        if material[p] == 1:  
            C[p] = ti.Matrix.zero(float, 3, 3)  
            v[p] = ti.Vector([0, 0, config["world"]["speed"]])  
            # v[p] = ti.Vector([0, 0, -200])  

        x[p] += dt * v[p]   
        dx_display, dy_display = config["world"]["display_shift"]["dx"], config["world"]["display_shift"]["dy"]
        scale_display = config["world"]["display_shift"]["scale"]
        # Handle display coordinates  
        x_2d[p] = [(x[p][1] * scale_display + dx_display), (x[p][2] * scale_display + dy_display)]  
        
        # Update F  
        F[p] = (ti.Matrix.identity(float, 3) + (dt * cached_C)) @ F[p] 
    # print("finished all steps")

@ti.func
def is_valid_vec(p):
    return p[0] >= 0 and p[0] < n_grid and p[1] >= 0 and p[1] < n_grid and p[2] >= 0 and p[2] < n_grid

@ti.func
def is_valid(i, j, k):
    return i >= 0 and i < n_grid and j >= 0 and j < n_grid and k >= 0 and k < n_grid
    
@ti.kernel
def initialize(data: ti.types.ndarray(),data_len: ti.i32):

    #initialize elastomer
    for i,j,k in ti.ndrange(num_l,num_w,num_h):
        m = i+j*num_l+k*num_l*num_w
        # offset for elastomer
        offest = ti.Vector([x_offset-l/2,y_offset-w/2,z_offset-h/2])
        # offest = ti.Vector([0,0,h/2])
        x[m] = ti.Vector([i,j,k])*particle_dis+offest
        x_2d[m] = [x[m][0], x[m][1]]
        v[m] = [0, 0, 0]
        material[m] = 0
        F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[m] = ti.Matrix.zero(float, 3, 3)

    #initialize indenter
    for i in ti.ndrange(data_len):
        m = i+num_l*num_w*num_h
        x[m] = ti.Vector([data[i,0],data[i,1],data[i,2]])
        x_2d[m] = [x[m][0], x[m][1]]
        v[m] = [0, 0, 0]
        material[m] = 1
        F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[m] = ti.Matrix.zero(float, 3, 3)

def save(p_xpos_list, p_ypos_list, p_zpos_list):
    x_ = x.to_numpy()

    p_xpos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h,0]
    p_ypos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h,1]
    p_zpos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h,2]

    p_xpos = np.reshape(p_xpos,(1,num_l*num_w))
    p_ypos = np.reshape(p_ypos,(1,num_l*num_w))
    p_zpos = np.reshape(p_zpos,(1,num_l*num_w))

    p_xpos_list = np.concatenate((p_xpos_list,p_xpos))
    p_ypos_list = np.concatenate((p_ypos_list,p_ypos))
    p_zpos_list = np.concatenate((p_zpos_list,p_zpos))

    return p_xpos_list, p_ypos_list, p_zpos_list

# taichi gui :L res should set as square
gui = ti.GUI("Explicit MPM rotate", res=config["world"]["resolution"], background_color=0x112F41)
colors = np.array([0x808080,0x00ff00,0xEEEEF0], dtype=np.uint32)

# initialization
initialize(data, np.shape(data)[0])

# step sim
while True:
    t_ti[None] += 1
    p_xpos_list, p_ypos_list, p_zpos_list = save(p_xpos_list, p_ypos_list, p_zpos_list)
    depth = np.min(p_zpos_list[-1,:])
    substep()
    if t_ti[None]>=30:
        #press 1mm
        gel_surface_h = z_offset+h/2
        if (depth-(gel_surface_h-indenting_depth))>-1e-2:
            if t_ti[None]%2==0:
                p_xpos_list, p_ypos_list, p_zpos_list = save(p_xpos_list, p_ypos_list, p_zpos_list)
        else:
            np.savez(args.dir_output + "/" + args.object + "/"+ suffix ,p_xpos_list=p_xpos_list, p_ypos_list=p_ypos_list, p_zpos_list=p_zpos_list)
            print("press saved")
            sys.exit()

    scale = config["world"]["display_scale"]
    gui.circles(x_2d.to_numpy()/scale, radius=1, color=colors[material.to_numpy()])
    gui.show() 

    print(t_ti[None])