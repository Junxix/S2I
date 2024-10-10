import os
import sys
from torchvision import transforms, datasets

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.constant import *
from dataset.dataset import RealworldValDataset

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),
        normalize,
    ])
    
    # val_dataset = ValDataset(hdf5_file = "./low_dim_select30.hdf5", transform=val_transform)
    val_dataset = RealworldDataset(dataset = './20%mixdata_sampled_20240320', transform=val_transform, save_mode='realworld')

    for idx in range(0, len(val_dataset)):
        image_data, demo_idx, small_demo_idx = val_dataset[idx]
        to_pil = transforms.ToPILImage()
        output_image = to_pil(image_data)

        output_image.save(f'./tmp/output_image_{demo_idx}_{small_demo_idx}.png')