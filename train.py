from __future__ import print_function
import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils.util import *
from utils.constant import *
from utils.data_augmentation import data_augmentation
from networks.resnet import SupConResNet
from losses import SupConLoss
from dataset.dataset import CustomDataset

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser('argument for training')

    # Training configurations
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='Save model frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=3000, help='Total training epochs')

    # Optimization configurations
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='Epochs where learning rate decays')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')

    # Model and dataset settings
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument("--dataset", type=str, default='./low_dim.hdf5', help="path to hdf5 dataset")
    parser.add_argument('--aug_path', type=str, default=None, help='Path to custom dataset')
    parser.add_argument("--save_mode", type=str, default='lowdim', choices=['image', 'lowdim', 'realworld'], help="choose the saving method")
    parser.add_argument('--size', type=int, default=128, help='Image size for RandomResizedCrop')

    # Data augmentation
    parser.add_argument("--total_images", type=int, default=100, help="total number of images to generate")
    parser.add_argument("--numbers", type=int, nargs='+', default=[0, 1, 2], help="list of numbers for processing")

    # Method and loss function configurations
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'], help='Contrastive learning method')
    parser.add_argument('--temp', type=float, default=0.01, help='Temperature for loss function')

    # Paths for saving model and tensorboard logs
    parser.add_argument('--model_path', type=str, default='./lowdim/models', help='Path to save model checkpoints')
    parser.add_argument('--tb_path', type=str, default='./lowdim/tensorboard', help='Path for tensorboard logs')

    # Other settings
    parser.add_argument('--cosine', action='store_true', help='Use cosine annealing learning rate schedule')
    parser.add_argument('--syncBN', action='store_true', help='Use synchronized Batch Normalization')
    parser.add_argument('--warm', action='store_true', help='Use warm-up for large batch training')

    opt = parser.parse_args()
    opt.lr_decay_epochs = list(map(int, opt.lr_decay_epochs.split(',')))

    opt.model_name = f'{opt.method}_{opt.model}_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}_temp_{opt.temp}_imgsize_{opt.size}'
    if opt.cosine:
        opt.model_name += '_cosine'
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name += '_warm'
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3) if opt.cosine else opt.learning_rate
        opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def set_loader(opt):
    """ Data loader for the training dataset """
    normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.8, 1.)),
        transforms.ToTensor(),
        normalize,
    ])

    
    train_dataset = CustomDataset(npy_file=opt.aug_path, transform=TwoCropTransform(train_transform))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader

def set_model(opt):
    """ Initialize model and loss function """
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # Optional synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda() if torch.cuda.device_count() > 1 else model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """ One epoch training loop """
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images, labels = images.cuda(), labels.cuda(non_blocking=True)

        bsz = labels.size(0)

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels) if opt.method == 'SupCon' else criterion(features)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})')

    return losses.avg

def main():
    """ Main function to train the model """
    opt = parse_option()
    data_augmentation(opt)

    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        start_time = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        print(f'Epoch {epoch}, Total Time {time.time() - start_time:.2f}, Loss {loss:.4f}, Learning Rate {optimizer.param_groups[0]["lr"]}')

        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
