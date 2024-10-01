import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
# from robomimic.envs.env_base import EnvBase, EnvType
from torchvision import transforms, datasets
# import robomimic.utils.obs_utils as ObsUtils
# import robomimic.utils.env_utils as EnvUtils
# import robomimic.utils.file_utils as FileUtils
# import matplotlib.pyplot as plt
# # from .extract_waypoints import trajectory_optimization
# from .trajectory_optimization import TrajectoryOptimizer

# DEFAULT_CAMERAS = {
#     EnvType.ROBOSUITE_TYPE: ["agentview"],
#     EnvType.IG_MOMART_TYPE: ["rgb"],
#     EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
# }

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
    def __init__(self, hdf5_file, transform=None):
        """
        Args:
            hdf5_file (string): HDF5 文件路径.
            transform (callable, optional): 可选的图像转换.
        """
        self.transform = transform
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
            print(marks)
        else:
            marks = list(range(small_demo['frame_start'], small_demo['frame_end']))

        self.save_marks(demo_idx, marks)


    def save_marks(self, demo_idx, marks):
        with h5py.File(self.hdf5_file, 'a') as f: 
            if f'data/{demo_idx}/marks' not in f:
                f.create_dataset(f'data/{demo_idx}/marks', data=marks)
            else:
                existing_marks = f[f'data/{demo_idx}/marks'][:]
                all_marks = np.unique(np.concatenate((existing_marks, marks)))
                all_marks.sort()
                del f[f'data/{demo_idx}/marks']  
                f.create_dataset(f'data/{demo_idx}/marks', data=all_marks)


    def transform_points(self, trajectory_points):
        camera_position = np.array([1.0, 0.0, 1.75])
        camera_rotation = np.array([
            [0.0, -0.70614724, 0.70806503],
            [1.0, 0.0, 0.0],
            [0.0, 0.70806503, 0.70614724]
        ])

        transformed_points = np.dot(trajectory_points - camera_position, camera_rotation)
        return transformed_points


    def generate_image(self, small_demo, save_mode="image"):
        trajectory_points = small_demo['trajectory_points']  # 假设从小 demo 中提取动作

        self.env.reset()
        self.env.reset_to(dict(states=small_demo['states'][0]))
        frame = self.env.render(mode="rgb_array", height=480, width=480, camera_name=self.render_image_names[0])  # 根据需要替换 camera_name
        
        image2 = Image.fromarray(frame)
        image_array = np.array(image2)

        if save_mode == 'lowdim':
            factor = 0
        elif save_mode == 'image':
            factor = 0.5
        elif save_mode == 'realworld':
            raise ValueError("The 'realworld' mode is not yet implemented.")
        else:
            raise ValueError(f"Unknown save_mode: {save_mode}")

        white_image = np.ones_like(image_array) * 255
        new_image_array = (image_array * factor + white_image * (1 - factor)).astype(np.uint8)

        image2 = Image.fromarray(new_image_array)
        transformed_points = self.transform_points(trajectory_points)  # 假设你有这个函数
    
        plt.clf()
        projected_points = transformed_points[:, :2]
        plt.plot(projected_points[:, 0], projected_points[:, 1], color='red', linewidth=5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('off')
        plt.xlim(-0.45, 0.45)
        plt.ylim(-0.5, 0.5)
        plt.gcf().set_size_inches(480/96, 480/96)
        plt.tight_layout()
        plt.savefig('./test_dataset/test.png', transparent=True, dpi=96)

        image1 = Image.open('./test_dataset/test.png')
        image2.paste(image1, (0, 0), image1)
        return image2

        

if __name__ == '__main__':
    mean =  "(0.4914, 0.4822, 0.4465)"
    std = "(0.2675, 0.2565, 0.2761)"
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dataset = ValDataset(hdf5_file = "/aidata/jingjing/data/robomimic/can/mh/low_dim/select30/low_dim_select30.hdf5", transform=val_transform)
    for idx in range(0, len(val_dataset)):
        image_data, demo_idx, small_demo_idx = val_dataset[idx]
        print(demo_idx)
        output_image.save(f'../test_dataset/output_image_{demo_idx}_{small_demo_idx}.png')