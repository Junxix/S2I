import os
import sys


import os
import json
import torch
import time
import psutil
import sys
import traceback
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
import logging
import random

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n': 
            self.level(message)

    def flush(self):
        pass

def setup_logging(log_file_path):
    with open(log_file_path, 'w'):
        pass

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    sys.stdout = LoggerWriter(logging.info)


def rollout_from_checkpoint(config, checkpoint_path, device,  seed,  video_dir, epoch):

    print(f"\n============= Performing Rollout for Seed {seed} and Epoch {epoch} =============")

    set_seed(seed)

    ObsUtils.initialize_obs_utils_with_config(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dataset_path = os.path.expanduser(config.train.data)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    envs = OrderedDict()
    env_names = [env_meta["env_name"]]
    if config.experiment.additional_envs is not None:
        env_names.extend(config.experiment.additional_envs)

    for env_name in env_names:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name,
            render=False,
            render_offscreen=config.experiment.render_video,
            use_image_obs=shape_meta["use_images"],
        )
        envs[env.name] = env
        print(envs[env.name])

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    
    model.load_state_dict(checkpoint["model"])

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
        obs_normalization_stats = trainset.get_obs_normalization_stats()
        
    rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

    num_episodes = config.experiment.rollout.n
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=envs,
        horizon=config.experiment.rollout.horizon,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
    )

    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]
        print("\nRollouts results for env {}:".format(env_name))
        print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    process = psutil.Process(os.getpid())
    mem_usage = int(process.memory_info().rss / 1000000)
    print("\nMemory Usage: {} MB\n".format(mem_usage))


def extract_epoch_from_filename(filename):

    try:
        base_name = os.path.basename(filename)
        epoch_str = base_name.split("model_epoch_")[1].split('_')[0].split(".pth")[0]
        return int(epoch_str)
    except (IndexError, ValueError):
        pass
    return None



def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # log_file_path = os.path.join(args.checkpoint_dir, "eval-log.txt")
    # setup_logging(log_file_path)

    checkpoint_dir = args.checkpoint_dir
    checkpoints = []
    
    for root, _, files in os.walk(args.checkpoint_dir):
        for file in files:
            if file.endswith(".pth"):
                epoch = extract_epoch_from_filename(file)
                if epoch is None:
                    print(f"Failed to extract epoch from filename: {file}")
                    continue
                checkpoint_path = os.path.join(root, file)
                checkpoints.append((epoch, checkpoint_path))

    checkpoints.sort(key=lambda x: x[0])

    for epoch, checkpoint_path in checkpoints:
        try:
            seed = int(checkpoint_path.split('seed')[-1].split('/')[0])
        except (ValueError, IndexError):
            print(f"Failed to extract seed from path: {checkpoint_path}")
            continue
        
        video_dir = os.path.join(os.path.dirname(checkpoint_path), "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        print(f"Testing checkpoint: {checkpoint_path} with seed: {seed}, saving videos to: {video_dir}")
        try:
            rollout_from_checkpoint(config, checkpoint_path, device=device, seed=seed, video_dir=video_dir, epoch=epoch)
        except Exception as e:
            print(f"Rollout failed for {checkpoint_path} with error:\n{e}\n\n{traceback.format_exc()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config JSON file")
    parser.add_argument("--algo", type=str, help="Algorithm name")
    parser.add_argument("--dataset", type=str, help="Dataset path")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint models")

    args = parser.parse_args()
    main(args)
