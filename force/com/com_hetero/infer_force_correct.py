import os
import torch
import numpy as np
import data_loader
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import utils.utils as utils
import pandas as pd
from tqdm import tqdm  

np.set_printoptions(precision=2,suppress=True)
torch.set_printoptions(sci_mode=False)  

def visual_predict(groundtruth, prediction, fig_save_dir):  
    """  
    Visualize the predictions vs ground truth on a scatter plot (Fx, Fy, Fz) with R^2.  
    """  
    plt.style.use("force/config/fig.mplstyle")  
    force_labels = [r"$F_x$", r"$F_y$", r"$F_z$"]  
    colors = ["darkturquoise", "darkorchid", "crimson"]  
    bounds = [(-5, 5, 4), (-5, 5, 4), (-20, 5, 10)]  # (min, max, step) for ticks in each axis  

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

    r2_save = []
    for i, ax in enumerate(axes):  
        ax.scatter(groundtruth[:, i], prediction[:, i], alpha=0.7, label=force_labels[i], c=colors[i])  
        xreg = np.linspace(*bounds[i][:2])  
        ax.plot(xreg, xreg, color="darkgrey", lw=1, linestyle="dashed")  
        r2 = r2_score(groundtruth[:, i], prediction[:, i]) 
        r2_save.append(r2)
        ax.text(bounds[i][0] + 1, bounds[i][1] - 2, f"$R^2={r2:.2f}$", fontsize=8, c=colors[i])  
        ax.set_xlabel("Groundtruth (N)")  
        ax.set_ylabel("Prediction (N)")  
        ax.set_xticks(np.arange(*bounds[i]))  
        ax.set_yticks(np.arange(*bounds[i]))  
        ax.legend(loc=0)  
    plt.tight_layout()  
    plt.savefig(fig_save_dir, dpi=600)  
    plt.close()  
    return r2_save


def to_normal(m, y):  
    """Reverse normalization for predictions, min_max"""  
    y = y * (m[1] - m[0]) + m[0]  
    return y  


def total_force(y):  
    """Compute total force."""  
    return torch.sqrt(torch.sum(torch.square(y),dim=1))  

def solve_root(polynomial,force):
    
    polynomial = polynomial - force  
    roots = np.roots(polynomial)    
    threshold_min, threshold_max = 0, 1.5
    real_roots = roots[np.isreal(roots)].real  # Keep only real roots  
    # filtered_roots = real_roots[(real_roots > threshold_min) & (real_roots < threshold_max)] 
    filtered_roots = real_roots
    print(f'roots:{filtered_roots}')
    return filtered_roots


def correct_force(predicted_force, modulus_func):
    predicted_force[:,2] = predicted_force[:,2] * (-1)
    down_idx_ori = np.diff(predicted_force[:,2])>0
    mask = np.insert(down_idx_ori,0,1)
    forces_corrected = []
    for is_down, force in zip(mask, predicted_force):
        if is_down:
            depth = solve_root(modulus_func["source_down"],force[2])
            ratio = modulus_func["target_down"](depth) / force[2]
            force_after = force * ratio
            force_after[2] = force_after[2] * (-1)
            forces_corrected.append(force_after)
        else:
            depth = solve_root(modulus_func["source_up"],force[2])
            ratio = modulus_func["target_up"](depth) / force[2]
            force_after = force * ratio
            force_after[2] = force_after[2] * (-1)
            forces_corrected.append(force_after)
    forces_corrected = np.array(forces_corrected).reshape(-1,3)
    print(f"ori force:{predicted_force}")
    print(f"corrected force:{forces_corrected}")
    return forces_corrected

def predict_force(model, global_min_max, args, inf_dir, target_data_loader, label):  
    """  
    Predict forces using the trained model and compute errors (MAEs and total force).  
    Save predictions, groundtruths, and errors for visualizations and analysis.  
    """  
    # File names for saving results  
    file_map = {  
        "source": {"error": "s_error.txt", "groundtruth": "s_groundtruth.npy", "prediction": "s_prediction.npy", "fig": "s.png", "r2":"s_r2.txt"},  
        "target": {"error": "t_error.txt", "groundtruth": "t_groundtruth.npy", "prediction": "t_prediction.npy", "fig": "t.png", "r2":"t_r2.txt"}  
    }  
    file_paths = {key: os.path.join(inf_dir, val) for key, val in file_map[label].items()} 

    if label == "target":
        source, target = args.task.split("-")
        modulus_func = {"source_down":np.poly1d(np.loadtxt(os.path.join(args.modulus,f"{source}_down.csv"), delimiter=",")), 
                        "source_up":np.poly1d(np.loadtxt(os.path.join(args.modulus,f"{source}_up.csv"), delimiter=",")),
                        "target_down":np.poly1d(np.loadtxt(os.path.join(args.modulus,f"{target}_down.csv"), delimiter=",")), 
                        "target_up":np.poly1d(np.loadtxt(os.path.join(args.modulus,f"{target}_up.csv"), delimiter=","))}

    # Load normalization statistics    
    global_min = torch.tensor((global_min_max['min'])).cuda()
    global_max = torch.tensor((global_min_max['max'])).cuda()
    normalization_stats = (global_min, global_max) 

    # Initialize metrics  
    criterion_reg = torch.nn.L1Loss(reduction="mean")  
    e_metrics = {key: utils.AverageMeter() for key in ["ex", "ey", "ez", "etf"]}  
    groundtruths, predictions = [], []  

    model.to(args.device).eval()  
    # Loop through test data  
    with torch.no_grad():  
        for img, force in tqdm(target_data_loader):  
            img = img.to(args.device)  
            groundtruth = force.to(args.device)  
            # Predict last frame only  
            predicted_force = model(img)  # (s, 1, 3)
            # print(predicted_force.shape)
            predicted_force = torch.reshape(predicted_force,(-1,3))
            predicted_force = to_normal(normalization_stats, predicted_force) 
            groundtruth = to_normal(normalization_stats, groundtruth) 
            #use correction
            if label == "target":
                predicted_force = correct_force(predicted_force.cpu().numpy(), modulus_func)
                predicted_force = torch.tensor(predicted_force).to(args.device)  

            groundtruth = torch.reshape(groundtruth,(-1,3))
            print(f"groundtruth:{groundtruth}")
            # predicted_force = to_normal(normalization_stats, predicted_force[-1,:])  
            # groundtruth = to_normal(normalization_stats, groundtruth[-1,:])  
            # Calculate per-axis and total force errors  
            e_metrics["ex"].update(criterion_reg(predicted_force[:,0], groundtruth[:,0]).item())  
            e_metrics["ey"].update(criterion_reg(predicted_force[:,1], groundtruth[:,1]).item())  
            e_metrics["ez"].update(criterion_reg(predicted_force[:,2], groundtruth[:,2]).item())  
            e_metrics["etf"].update(criterion_reg(total_force(predicted_force), total_force(groundtruth)).item())  

            # Store groundtruths and predictions  
            groundtruths.append(groundtruth.cpu().numpy())  
            predictions.append(predicted_force.cpu().numpy())  

    # Aggregate results and save  
    groundtruth_array = np.vstack(groundtruths)  
    prediction_array = np.vstack(predictions)  
    
    r2 = visual_predict(groundtruth_array, prediction_array, file_paths["fig"])  # Save scatter plot  
    errors = [e_metrics[key].avg for key in ["ex", "ey", "ez", "etf"]]  
    np.savetxt(file_paths["error"], errors, delimiter=",", fmt="%.4f")  
    np.savetxt(file_paths["r2"], r2, delimiter=",", fmt="%.2f")  
    np.save(file_paths["groundtruth"], groundtruth_array)  
    np.save(file_paths["prediction"], prediction_array)  

    # Print errors  
    print(f"{label}: ex {errors[0]:.3f} ey {errors[1]:.3f} ez {errors[2]:.3f} etf {errors[3]:.3f}")  
