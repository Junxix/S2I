import os
import cv2
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from .trajectory_optimization import TrajectoryOptimizer
from utils.realworld_utils import *
from utils.constant import *


class CustomDataset(Dataset):
    def __init__(self, npy_file, transform=None):
        """
        Args:
            npy_file (string): 
            transform (callable, optional): 
        """
        self.transform = transform
        self.data = np.load(npy_file, allow_pickle=True).item()
        self.inputs = self.data['images']
        self.labels = self.data['labels']
        
        if len(self.inputs) != len(self.labels):
            raise ValueError(f"Length mismatch: inputs({len(self.inputs)}) and labels({len(self.labels)})")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]
        if isinstance(input_data, np.ndarray):
            input_data = Image.fromarray(input_data)

        if self.transform:
            input_data = self.transform(input_data)
        
        return input_data, label




class ValDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, save_mode=None):
        """
        Args:
            hdf5_file (string)
            transform (callable, optional)
        """
        self.transform = transform
        self.save_mode = save_mode
        self.hdf5_file = hdf5_file
        self.data = h5py.File(hdf5_file, 'r')

        self.demos = list(self.data["data"].keys())
        self.small_demos = {}
        self.mapping = {}
        self._init_env()
        self.optimizer = TrajectoryOptimizer(self.env, real_world=False)

        self._split_demos()
        self.data.close()


    def _init_env(self):
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.hdf5_file)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        self.render_image_names = DEFAULT_CAMERAS[env_type]

        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
        self.env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
        self.camera_name = DEFAULT_CAMERAS[env_type][0]
        self.extrinsic_matrix = self.env.get_camera_extrinsic_matrix(camera_name)
        self.camera_position = self.extrinsic_matrix[:3, 3]
        self.camera_rotation = self.extrinsic_matrix[:3, :3]
        
    def _split_demos(self):
        for demo_idx in self.demos:
            actions = self.data[f'data/{demo_idx}/actions'][()]
            states = self.data[f'data/{demo_idx}/states'][()]
            trajectory_points = self.data[f'data/{demo_idx}/obs/robot0_eef_pos'][()]
            frames = self._get_frames(actions)
            small_demos = self._split_into_small_demos(actions, states, trajectory_points, frames)

            self.small_demos[demo_idx] = small_demos
            self.mapping[demo_idx] = list(range(len(small_demos)))  

    def _get_frames(self, actions):
        frames = [0, len(actions) - 1]
        for i in range(len(actions) - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                frames.append(i)
        frames.sort()

        merged_frames = [frames[0]]
        for i in range(1, len(frames)):
            if frames[i] - merged_frames[-1] < 15: 
                merged_frames.pop()
            merged_frames.append(frames[i])

        merged_frames.sort()
        return merged_frames

    def _split_into_small_demos(self, actions, states, trajectory_points, frames):
        small_demos = []
        for i in range(len(frames) - 1):
            start, end = frames[i], frames[i + 1]
            small_demos.append({
                'actions': actions[start:end],
                'states': states[start:end],
                'trajectory_points': trajectory_points[start:end],
                'frame_start': frames[i],
                'frame_end': frames[i+1]
            })
        return small_demos

    def __len__(self):
        return sum(len(small_demo) for small_demo in self.small_demos.values())
    
    def __getitem__(self, idx):
        demo_idx, small_demo_idx = self._find_small_demo_index(idx)
        small_demo = self.small_demos[demo_idx][small_demo_idx]

        positive_image = self.generate_image(small_demo)

        if self.transform:
            positive_image = self.transform(positive_image)

        return positive_image, demo_idx, small_demo_idx

    def _find_small_demo_index(self, idx):
        for demo_idx, small_demos in self.small_demos.items():
            if idx < len(small_demos):
                return demo_idx, idx
            idx -= len(small_demos)
        raise IndexError("Index out of range.")

    def _save_marks(self, demo_idx, marks):
        with h5py.File(self.hdf5_file, 'a') as f: 
            if f'data/{demo_idx}/marks' not in f:
                f.create_dataset(f'data/{demo_idx}/marks', data=marks)
            else:
                existing_marks = f[f'data/{demo_idx}/marks'][:]
                all_marks = np.unique(np.concatenate((existing_marks, marks)))
                all_marks.sort()
                del f[f'data/{demo_idx}/marks']  
                f.create_dataset(f'data/{demo_idx}/marks', data=all_marks)


    def visualize_image(self,idx):
        demo_idx, small_demo_idx = self._find_small_demo_index(idx)
        small_demo = self.small_demos[demo_idx][small_demo_idx]

        positive_image = self.generate_image(small_demo)
        return positive_image


    def perform_optimization(self, idx, flag=True):
        demo_idx, small_demo_idx = self._find_small_demo_index(idx)
        small_demo = self.small_demos[demo_idx][small_demo_idx]
        if flag:
            marks = self.optimizer.optimize_trajectory(small_demo, demo_idx, small_demo_idx,three_dimension=True)
        else:
            marks = list(range(small_demo['frame_start'], small_demo['frame_end']))

        self._save_marks(demo_idx, marks)

    def generate_image(self, small_demo, save_mode="image"):
        trajectory_points = small_demo['trajectory_points'] 

        self.env.reset()
        self.env.reset_to(dict(states=small_demo['states'][0]))
        frame = self.env.render(mode="rgb_array", height=480, width=480, camera_name=self.render_image_names[0])  
        
        image = Image.fromarray(frame)
        factor = get_save_mode_factor(save_mode=self.save_mode)
        image = apply_image_filter(image, factor)

        transformed_points = np.dot(trajectory_points - self.camera_position, self.camera_rotation)
        return plot(transformed_points, image)


class RealworldDataset(Dataset):
    def __init__(self, dataset, transform=None, save_mode=None):
        """
        Args:
            hdf5_file (string)
            transform (callable, optional)
        """
        self.transform = transform
        self.calib_dir = check_directory_exists(os.path.join(dataset, "calib"))
        self.save_mode = save_mode
        self.root_dir = os.path.join(dataset, "train")
        subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.subdirs = sorted(subdirs, key=lambda x: int(x.split('_scene_')[1].split('_')[0]))
        self.small_demos = {}
        self.mapping = {}

        self.optimizer = TrajectoryOptimizer(env=None, real_world=True, calib_dir=self.calib_dir)
        self._split_demos()

    def _split_demos(self):
        for ind in range(len(self.subdirs)-20):
            color_path =  os.path.join(self.root_dir, self.subdirs[ind], CAMERA_NAME, 'color')
            file_paths, trajectory_points, gripper_command = load_demo_files(self.root_dir, self.subdirs, ind)
            frames = realworld_change_indices(gripper_command)
            small_demos = self._split_into_small_demos(file_paths, trajectory_points, frames, color_path)

            self.small_demos[ind] = small_demos
            self.mapping[ind] = list(range(len(small_demos)))  

    def _split_into_small_demos(self, file_paths, trajectory_points, frames, color_path):
        small_demos = []
        for i in range(len(frames) - 1):
            start, end = frames[i], frames[i + 1]
            small_demos.append({
                'states': file_paths[start:end],
                'trajectory_points': trajectory_points[start:end],
                'frame_start': frames[i],
                'frame_end': frames[i+1],
                'color_path': color_path
            })
        return small_demos

    def _find_small_demo_index(self, idx):
        for demo_idx, small_demos in self.small_demos.items():
            if idx < len(small_demos):
                return demo_idx, idx
            idx -= len(small_demos)
        raise IndexError("Index out of range.")
    
    def _generate_image(self, small_demo):
        trajectory_points = small_demo['trajectory_points']

        transformed_points = translate_points(self.calib_dir, trajectory_points)
        img_path = os.path.join(small_demo['color_path'], small_demo['states'][0])            
        image = cv2.imread(img_path)

        factor = get_save_mode_factor(save_mode=self.save_mode)
        image = apply_image_filter(image, factor)
        image = np.array(image)

        prev_point = None
        for point in transformed_points[:]:
            cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
            if prev_point is not None:
                cv2.line(image, prev_point, point, (0, 0, 255), thickness=4)
            prev_point = point
        return image

    def __len__(self):
        return sum(len(small_demo) for small_demo in self.small_demos.values())
    
    def __getitem__(self, idx):
        demo_idx, small_demo_idx = self._find_small_demo_index(idx)
        small_demo = self.small_demos[demo_idx][small_demo_idx]
        positive_image = self._generate_image(small_demo)

        positive_image = self.transform(Image.fromarray(positive_image)) if self.transform and isinstance(positive_image, np.ndarray) else positive_image

        return positive_image, demo_idx, small_demo_idx

    def _save_marks(self, demo_idx, marks, color_path):
        npy_file_path = os.path.join(color_path, f'marks_{demo_idx}.npy')
        if os.path.exists(npy_file_path):
            existing_marks = np.load(npy_file_path)
            all_marks = np.unique(np.concatenate((existing_marks, marks)))
        else:
            all_marks = np.unique(marks)
        
        all_marks.sort() 
        np.save(npy_file_path, all_marks)

    def perform_optimization(self, idx, flag=True):
        demo_idx, small_demo_idx = self._find_small_demo_index(idx)
        small_demo = self.small_demos[demo_idx][small_demo_idx]
        if flag:
            marks = self.optimizer.optimize_trajectory(small_demo, demo_idx, small_demo_idx,three_dimension=True)
        else:
            marks = list(range(small_demo['frame_start'], small_demo['frame_end']))

        self._save_marks(demo_idx, marks, small_demo['color_path'])

