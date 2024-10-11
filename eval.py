import os
import shutil
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm
from utils.util import TwoCropTransform, AverageMeter
from networks.resnet import SupConResNet
from dataset.dataset import CustomDataset, ValDataset
from utils.constant import *

def parse_option():
    parser = argparse.ArgumentParser('Argument for training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument("--save_mode", type=str, default='lowdim', choices=['image', 'lowdim', 'realworld'], help="choose the saving method")
    parser.add_argument('--train_data_folder', type=str, default='./lowdim_samples.npy', help='path to custom dataset')
    parser.add_argument('--val_data_folder', type=str, default='./low_dim.hdf5', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--ckpt', type=str, default='./ckpt_epoch_2000.pth',
                        help='path to pre-trained model')
    return parser.parse_args()

def dist_metric(x, y):
    return torch.norm(x - y).item()

def calculate_label(dist_list, k):
    top_k_weights = torch.nn.functional.softmax(torch.tensor([d[0] for d in dist_list[:k]]) * -1, dim=0)
    action = sum(weight * dist_list[i][1] for i, weight in enumerate(top_k_weights))
    return action

def clear_folders_if_not_empty(folders):
    for folder in folders:
        if os.path.exists(folder) and os.listdir(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)

def calculate_nearest_neighbors(query_embedding, train_dataset, train_labels, k):
    dist_list = [(dist_metric(torch.from_numpy(query_embedding), torch.from_numpy(train_dataset[i])), train_labels[i]) for i in range(len(train_dataset))]
    dist_list.sort(key=lambda tup: tup[0])
    return calculate_label(dist_list, k)

def set_loader(opt):
    normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CustomDataset(npy_file=opt.train_data_folder, transform=train_transform)
    val_dataset = ValDataset(hdf5_file=opt.val_data_folder, transform=val_transform, save_mode = opt.save_mode)
    return train_dataset, val_dataset

def set_model(opt):
    model = SupConResNet(name=opt.model)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError('This code requires GPU')
    return model

def get_embeddings(train_dataset, model):
    model.eval()
    embeddings, labels = [], []
    for idx in range(len(train_dataset)):
        image, label = train_dataset[idx]
        image = image.unsqueeze(0).cuda(non_blocking=True)
        with torch.no_grad():
            features = model.encoder(image).flatten(start_dim=1)
        embeddings.append(features.cpu().numpy())
        labels.append(label)
    return np.concatenate(embeddings), np.array(labels)

def classifier(val_dataset, train_dataset, train_labels, model, neighbors_num):
    device = next(model.parameters()).device
    dest_folders = ['./test_dataset/negative/', './test_dataset/positive/']
    clear_folders_if_not_empty(dest_folders)

    for idx in tqdm(range(len(val_dataset))):
        image_data, demo_idx, small_demo_idx = val_dataset[idx]
        image_data = image_data.unsqueeze(0).to(device)
        
        with torch.no_grad():
            val_embedding = model.encoder(image_data).cpu().numpy()

        label = calculate_nearest_neighbors(val_embedding, train_dataset, train_labels, neighbors_num)
        flag = label < 0.5
        val_dataset.perform_optimization(idx, flag=flag)
        
        folder = dest_folders[0] if flag else dest_folders[1]
        image_path = f"{folder}/output_image_{demo_idx}_{small_demo_idx}.png"
        val_dataset.visualize_image(idx).save(image_path)

def main():
    opt = parse_option()
    train_dataset, val_dataset = set_loader(opt)
    model = set_model(opt)
    train_embeddings, train_labels = get_embeddings(train_dataset, model)
    classifier(val_dataset, train_embeddings, train_labels, model, neighbors_num=64)

if __name__ == '__main__':
    main()
