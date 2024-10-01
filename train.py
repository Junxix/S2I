from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util import TwoCropTransform, AverageMeter, adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model
from networks.resnet_big import SupConResNet
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
    parser.add_argument('--mean', type=str, default="(0.4914, 0.4822, 0.4465)", help='Mean for normalization')
    parser.add_argument('--std', type=str, default="(0.2675, 0.2565, 0.2761)", help='Standard deviation for normalization')
    parser.add_argument('--data_folder', type=str, default=None, help='Path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='Image size for RandomResizedCrop')

    # Method and loss function configurations
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'], help='Contrastive learning method')
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for loss function')

    # Paths for saving model and tensorboard logs
    parser.add_argument('--model_path', type=str, default='/aidata/jingjing/chkpts/SupContrast/test/can/image-background-size128/models', help='Path to save model checkpoints')
    parser.add_argument('--tb_path', type=str, default='/aidata/jingjing/chkpts/SupContrast/test/can/image-background-size128/tensorboard', help='Path for tensorboard logs')

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
    mean = eval(opt.mean)
    std = eval(opt.std)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.8, 1.)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CustomDataset(npy_file=opt.data_folder, transform=TwoCropTransform(train_transform))

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

        # Compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels) if opt.method == 'SupCon' else criterion(features)
        losses.update(loss.item(), bsz)

        # Backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure batch time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if (idx + 1) % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})')

    return losses.avg

def main():
    """ Main function to train the model """
    opt = parse_option()

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
