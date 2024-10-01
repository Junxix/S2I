import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from dataset.trajectory_optimization import TrajectoryOptimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory optimization with real-world option")
    parser.add_argument('--real_world', action='store_true', help='Use real world parameters')
    return parser.parse_args()

def main():
    args = parse_args()
    env = None  
    optimizer = TrajectoryOptimizer(env, real_world=args.real_world)

    small_demo = {'trajectory_points': np.random.rand(10, 3), 'frame_start': 0}
    optimized_demo = optimizer.optimize_trajectory(small_demo, demo_idx=0, small_demo_idx=0, three_dimension=True)

if __name__ == '__main__':
    main()
