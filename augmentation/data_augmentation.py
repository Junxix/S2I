import os
import numpy as np
import h5py
from tqdm import tqdm
from trajectory_utils import find_all_change_indices, slice_trajectory_and_states
from image_generation import TrajectoryRenderer, PictureGenerator
import argparse

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

def data_augmentation(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_type = EnvUtils.get_env_type(env_meta=env_meta)
    render_image_names = DEFAULT_CAMERAS[env_type]

    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    samples = {'images': [], 'end_images': [], 'labels': []}

    total_demos = len(args.numbers)  
    total_images_per_demo = args.total_images // total_demos  

    renderer = TrajectoryRenderer(env, render_image_names[0])
    generator = PictureGenerator(renderer, samples, save_mode=args.save_mode)

    for ind in args.numbers:
        ep = demos[ind]
        states = f[f"data/{ep}/states"][()]
        trajectory_points = f[f"data/{ep}/obs/robot0_eef_pos"][()]
        actions = f[f"data/{ep}/actions"][()]

        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f[f"data/{ep}"].attrs["model_file"]
        env.reset()
        env.reset_to(initial_state)

        change_indices = find_all_change_indices(actions)
        trajectory_slices, state_slices = slice_trajectory_and_states(trajectory_points, states, change_indices)
        total_images_per_slice = total_images_per_demo // len(trajectory_slices)
        
        for i, (traj_slice, state_slice) in enumerate(zip(trajectory_slices, state_slices)):
            generator.generate_positive_picture(traj_slice, ind, num_images=total_images_per_slice)
            generator.generate_negative_picture(traj_slice, ind, num_images=total_images_per_slice)

    output_file = os.path.join(args.video_path_ori, f'{args.save_mode}_samples.npy')
    np.save(output_file, samples)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='/aidata/jingjing/robomimic/can/mh/low_dim_v141_saved.hdf5', help="path to hdf5 dataset")
    
    parser.add_argument("--video_path_ori", type=str, default='./tmp/', help="render trajectories to this video file path")

    parser.add_argument("--save_mode", type=str, default='lowdim', choices=['image', 'lowdim', 'realworld'], help="choose the saving method")

    parser.add_argument("--total_images", type=int, default=300, help="total number of images to generate")
    parser.add_argument("--numbers", type=int, nargs='+', default=[0, 1, 3], help="list of numbers for processing")

    args = parser.parse_args()

    data_augmentation(args)
