import matplotlib.pyplot as plt
import numpy as np
import argparse

from utils.realworld_utils import *
from utils.constant import *

class TrajectoryOptimizer:
    def __init__(self, env, real_world=False, calib_dir=None):
        self.env = env
        self.real_world = real_world
        self.calib_dir=calib_dir

    def calculate_vector(self, point1, point2):
        return point2 - point1

    def calculate_angle(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        cos_theta = dot_product / (norm_vector1 * norm_vector2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    def translate_points(self, trajectory_points):
        if self.real_world:
            transformed_points = translate_points(self.calib_dir, trajectory_points)
        else:
            extrinsic_matrix = self.env.get_camera_extrinsic_matrix('camera_name')
            camera_position = extrinsic_matrix[:3, 3]
            camera_rotation = extrinsic_matrix[:3, :3]
            transformed_points = np.dot(trajectory_points - camera_position, camera_rotation)

        projected_points = transformed_points[:, :2]
        return projected_points

    def calculate_new_points(self, points, three_dimension=False):
        if three_dimension:
            trajectory_points = points
        else:
            trajectory_points = self.translate_points(points)

        tolerance = np.max(np.linalg.norm(trajectory_points[1:] - trajectory_points[:-1], axis=1)) * 2
        selected_points = [trajectory_points[0]]  
        selected_indices = [0] 
        current_index = 0

        while current_index < len(trajectory_points) - 1:
            if all(trajectory_points[current_index] == trajectory_points[-1]):
                break

            distances = []
            for i in range(len(trajectory_points)):
                if not any(np.array_equal(trajectory_points[i], point) for point in selected_points):
                    distance = np.linalg.norm(trajectory_points[current_index] - trajectory_points[i])
                    distances.append((distance, i))

            sorted_distances = sorted(distances, key=lambda x: x[0])

            angle_candidates = []
            for distance, i in sorted_distances:
                nearest_vector = self.calculate_vector(trajectory_points[current_index], trajectory_points[i])
                goal_vector = self.calculate_vector(trajectory_points[current_index], trajectory_points[-1])
                angle = self.calculate_angle(nearest_vector, goal_vector)
                angle_candidates.append((angle, i))

                if distance > tolerance:
                    sorted_angles = sorted(angle_candidates, key=lambda x: x[0])
                    selected_points.append(trajectory_points[sorted_angles[0][1]])
                    selected_indices.append(sorted_angles[0][1])
                    current_index = sorted_angles[0][1]
                    break

                if angle < np.radians(DELTA_THETA):
                    selected_points.append(trajectory_points[i])
                    selected_indices.append(i)
                    current_index = i
                    break

        return selected_indices

    def optimize_trajectory(self, small_demo, demo_idx, small_demo_idx, three_dimension=False):
        trajectory_points = small_demo['trajectory_points']
        frame_start = small_demo['frame_start']

        small_demo_marks = self.calculate_new_points(np.array(trajectory_points), three_dimension)
        demo_marks = [mark + frame_start for mark in small_demo_marks]

        return demo_marks

