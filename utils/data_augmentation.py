import os
import numpy as np
import h5py
from tqdm import tqdm
from .realworld_utils import *
from .constant import *
from .image_generation import TrajectoryRenderer, PictureGenerator
import argparse

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

def process_slices(generator, trajectory_slices, color_path_slices, ind, total_images_per_slice):
    for i, (traj_slice, state_slice) in enumerate(zip(trajectory_slices, color_path_slices)):
        generator.generate_positive_picture(traj_slice, state_slice, ind, num_images=total_images_per_slice)
        generator.generate_negative_picture(traj_slice, state_slice, ind, num_images=total_images_per_slice)


def data_augmentation_realworld(args):
    calib_dir = check_directory_exists(os.path.join(args.dataset, "calib"))

    root_dir = os.path.join(args.dataset, "train")
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    sorted_subdirs = sorted(subdirs, key=lambda x: int(x.split('_scene_')[1].split('_')[0]))
    samples = {'images': [], 'end_images': [], 'labels': []}

    total_demos = len(args.numbers)  
    total_images_per_demo = args.total_images // total_demos  

    for ind in args.numbers:
        path = os.path.join(root_dir, sorted_subdirs[ind], CAMERA_NAME, 'color')
        renderer = TrajectoryRenderer(env=None, camera_name=None, save_mode='realworld', calib_dir=calib_dir, root_dir=path)
        generator = PictureGenerator(renderer, samples, save_mode=args.save_mode)

        file_paths, trajectory_points, gripper_command = load_demo_files(root_dir, sorted_subdirs, ind)

        change_indices = realworld_change_indices(gripper_command)
        trajectory_slices, color_path_slices = realworld_slice(trajectory_points, file_paths, change_indices)
        total_images_per_slice = total_images_per_demo // len(trajectory_slices)
        process_slices(generator, trajectory_slices, color_path_slices, ind, total_images_per_slice)
    
    np.save(args.aug_path, samples)


def data_augmentation_robomimic(args):
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
        generator.renderer.env.reset()
        generator.renderer.env.reset_to(initial_state)

        change_indices = find_all_change_indices(actions)
        trajectory_slices, state_slices = slice_trajectory_and_states(trajectory_points, states, change_indices)
        total_images_per_slice = total_images_per_demo // len(trajectory_slices)
        process_slices(generator, trajectory_slices, state_slices, ind, total_images_per_slice)

    np.save(args.aug_path, samples)
    f.close()


def data_augmentation(args):
    if args.save_mode == 'realworld':
        data_augmentation_realworld(args)
    else:
        data_augmentation_robomimic(args)

