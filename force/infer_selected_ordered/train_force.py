import os  
import torch  
import json
import configargparse  
import data_loader  
import numpy as np  
import random  
from model import TemporalForce  
from tqdm import tqdm  
from datetime import datetime  
from utils.utils import str2bool, AverageMeter  
from infer_force import predict_force  


def get_parser():  
    """Parse configuration arguments."""  
    parser = configargparse.ArgumentParser(  
        description="TransForce Config Parser",  
        config_file_parser_class=configargparse.YAMLConfigFileParser,  
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,  
    )  
    # Config file  
    parser.add_argument("--config", is_config_file=True, default="/scratch_tmp/users/k23058530/project/genforce/src/force/config/marker.yaml")  
    parser.add_argument('--task', type=str, default='GelSight1_GelTip2') 
    parser.add_argument('--unseen', type=str, nargs="+", default=[]) 
    parser.add_argument('--test_unseen', type=str2bool, default='False') 
    

    parser.add_argument('--infer_point', type=str, default="moon") 
    parser.add_argument('--infer_indenter', type=str, default="30") 

    # Data loader  
    parser.add_argument('--sf_filter', type=str, default="/scratch_tmp/users/k23058530/project/genforce/src/force/config/TacTip2_Normal>1.csv")  
    parser.add_argument('--tf_filter', type=str, default="/scratch_tmp/users/k23058530/project/genforce/src/force/config/TacTip2_Normal>1.csv")  
    parser.add_argument('--src_img', type=str, default="dataset/1")  
    parser.add_argument('--src_force', type=str, default="dataset/1")  
    parser.add_argument('--tar_img', type=str, default="dataset/1")  
    parser.add_argument('--tar_force', type=str, default="dataset/1")   
    parser.add_argument('--min_max', type=str, default="temp/min_max_m.npy")  
    parser.add_argument('--validation_split', type=float, default=0.2)  
    # parser.add_argument('--unseen', action="append")    

    # Training  
    parser.add_argument('--train', type=str2bool, default='True')  
    parser.add_argument("--seed", type=int, default=0)  
    parser.add_argument('--num_workers', type=int, default=1)  
    parser.add_argument('--checkpoint', type=str, default='/scratch_tmp/users/k23058530/project/genforce/output/force/GelSight1_GelTip2/model.pth')  
    parser.add_argument('--batch_size', type=int, default=8)  
    parser.add_argument('--n_epoch', type=int, default=20)  
    parser.add_argument('--out_dim', type=int, default=3)  
    parser.add_argument('--early_stop', type=int, default=20)  
    parser.add_argument('--lr', type=float, default=1e-3)  
    parser.add_argument('--momentum', type=float, default=0.9)  
    parser.add_argument('--weight_decay', type=float, default=5e-4)  
    parser.add_argument('--lr_gamma', type=float, default=3e-4)  
    parser.add_argument('--lr_decay', type=float, default=0.75)  
    parser.add_argument('--pth', type=str, default='epoch_{}.pth')  
    parser.add_argument('--log_train', type=str, default='loss_train.csv')  
    parser.add_argument('--log_test', type=str, default='loss_test.csv')  

    # Testing  
    parser.add_argument('--save_dir', type=str, default='force_prediction/{}')  
    parser.add_argument('--train_use_checkpoint', type=str2bool, default='False')  
    parser.add_argument('--draw_force', type=str2bool, default='True')  
    parser.add_argument('--draw_use_checkpoint', type=str2bool, default='False')  
    parser.add_argument('--draw_save_name', type=str, default='force.png')   

    return parser  


def set_random_seed(seed=0):  
    """Set random seed for reproducibility."""  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def load_data(args, min_max, shuffle=True, domain="source"):  
    """Load dataset for training or testing."""  
    if domain == "source" :
        img_dir = args.src_img    # xxx/images
        force_dir = args.src_force  # xxx/forces
        force_filter_list = args.sf_filter
    elif domain == "target":
        img_dir = args.tar_img    # xxx/images
        force_dir = args.tar_force  # xxx/forces
        force_filter_list = args.tf_filter
    else:
        raise RuntimeError("Please input a domain")
    return data_loader.create_dataloader(img_dir, force_dir, force_filter_list, min_max, args, shuffle=shuffle)  


def get_model(args):  
    """Initialize the model."""  
    return TemporalForce().to(args.device)  


def get_optimizer(model, args):  
    """Set up optimizer."""  
    params = model.get_parameters(initial_lr=args.lr)  
    return torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)  


def get_scheduler(optimizer, args):  
    """Set up learning rate scheduler."""  
    return torch.optim.lr_scheduler.LambdaLR(  
        optimizer, lambda x: (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)  
    )  


def make_save_dir(args):  
    """Create a directory for saving results."""  
    save_dir = args.save_dir.format(args.task)  
    os.makedirs(save_dir, exist_ok=True)  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    save_dir = os.path.join(save_dir, timestamp)  
    os.makedirs(save_dir, exist_ok=True)  
    return save_dir  


def to_normal(m, y):  
    """Reverse normalization for predictions, min_max"""  
    y = y * (m[:, 1] - m[:, 0]) + m[:, 0]  
    return y  


def total_force(y):  
    """Compute total force."""  
    return torch.sqrt(torch.sum(torch.square(y), dim=1))  


# def test(model, loader, args):  
#     """Evaluate the model."""  
#     min_max = np.load(args.min_max)  
#     model.eval()  
#     errors = {axis: AverageMeter() for axis in ['x', 'y', 'z', 'tf']}  
#     criterion = torch.nn.L1Loss()  

#     with torch.no_grad():  
#         for img, force in loader:  
#             img = img.permute(1, 0, 2, 3, 4).to(args.device)  # (s, b, c, H, W)
#             groundtruth = force.permute(1, 0, 2).to(args.device)  
#             predictions = model(img)  # (s, b, 3)

#             predictions = to_normal(min_max, predictions[-1])  # (b,3)
#             groundtruth = to_normal(min_max, groundtruth[-1])  # (b,3)

#             for i, axis in enumerate(['x', 'y', 'z']):  
#                 errors[axis].update(criterion(predictions[:, i], groundtruth[:, i]).item())  
#             errors['tf'].update(criterion(total_force(predictions), total_force(groundtruth)).item())  

#     return [errors[axis].avg for axis in ['x', 'y', 'z', 'tf']]  


def train(loader, model, optimizer, scheduler, args, log_path, checkpoint_path):  
    """Train the model."""  
    best_loss = float('inf')  
    stop_counter = 0  
    log_train = []  
    reg_mae = torch.nn.L1Loss(reduction='mean')  
    model.train()  
    for epoch in tqdm(range(1, args.n_epoch + 1), desc="Training Epochs"):  
        reg_loss_avg = AverageMeter()  
        for src_img, src_force in tqdm(loader, desc=f"Epoch {epoch}", leave=False):  
            src_img = src_img.to(args.device)  
            src_force = src_force.to(args.device)  
            predictions = model(src_img)   #(s, b, 3)  
            loss = reg_mae(predictions, src_force) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()  
            reg_loss_avg.update(loss.item())  
        log_train.append(reg_loss_avg.avg)  
        np.savetxt(log_path, np.array(log_train), delimiter=',', fmt='%.3f')  
        print('\nEpoch: [{:2d}/{}], reg_loss: {:.3f}\n'.format(epoch, args.n_epoch, reg_loss_avg.avg))
        torch.save(model.state_dict(), checkpoint_path.format(epoch))  
        if reg_loss_avg.avg < best_loss:  
            best_loss = reg_loss_avg.avg  
            stop_counter = 0  
        else:  
            stop_counter += 1  
        if stop_counter >= args.early_stop:  
            print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")  
            break  


def main():  
    parser = get_parser()  
    args = parser.parse_args()  
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    with open(args.min_max, "r") as file:  
        min_max_dict = json.load(file) 

    set_random_seed(args.seed)  
    save_dir = make_save_dir(args)  
    log_train_path = os.path.join(save_dir, args.log_train)  
    # checkpoint_save_path = os.path.join(save_dir, args.pth)  
    # source_loader = load_data(args, min_max_dict)  
    args.save_dir = save_dir

    print(args)
    np.savetxt(os.path.join(save_dir,'run_config.txt'),np.array(args).reshape(1,-1),fmt="%s")
    print('*'*200)
    # print("len(source): {:}".format(len(source_loader)))
    print("Task: {}".format(args.task))

    # model = get_model(args)
    # if args.train_use_checkpoint:  
    #     model.load_state_dict(torch.load(args.checkpoint))

    # optimizer = get_optimizer(model, args)  
    # scheduler = get_scheduler(optimizer, args)  

    if args.train:  
        train(source_loader, model, optimizer, scheduler, args, log_train_path, checkpoint_save_path)  

    if args.draw_force:  
        if args.draw_use_checkpoint:
            model = get_model(args) 
            model.load_state_dict(torch.load(args.checkpoint))
        setattr(args,"batch_size",1)
        target_data = load_data(args, min_max_dict, shuffle=False,domain="target")
        predict_force(model, min_max_dict, args, save_dir, target_data, "target")

    print(f"Task {args.task} completed.")  


if __name__ == "__main__":  
    main()