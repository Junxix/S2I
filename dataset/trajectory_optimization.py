import matplotlib.pyplot as plt
import numpy as np
import argparse

class TrajectoryOptimizer:
    def __init__(self, env, real_world=False):
        self.env = env
        self.real_world = real_world

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
            extrinsic_matrix = self.load_real_world_parameters()
        else:
            extrinsic_matrix = self.env.get_camera_extrinsic_matrix('camera_name')

        camera_position = extrinsic_matrix[:3, 3]
        camera_rotation = extrinsic_matrix[:3, :3]

        transformed_points = np.dot(trajectory_points - camera_position, camera_rotation)
        projected_points = transformed_points[:, :2]
        return projected_points

    def load_real_world_parameters(self):
        # realworld parameter
        calib_root_dir = '/aidata/jingjing/dataset/realdata_sampled_20240320/1710897318975'
        tcp_file = os.path.join(calib_root_dir, 'tcp.npy')
        extrinsics_file = os.path.join(calib_root_dir, 'extrinsics.npy')
        intrinsics_file = os.path.join(calib_root_dir, 'intrinsics.npy')

        tcp = np.load(tcp_file)
        extrinsics = np.load(extrinsics_file, allow_pickle=True).item()
        intrinsics = np.load(intrinsics_file, allow_pickle=True).item()
        position = tcp[:3]
        quaternion = tcp[3:]

        M_end_to_base = self.create_transformation_matrix(position, quaternion)
        M_cam0433_to_end = np.array([[0, -1, 0, 0],
                                     [1, 0, 0, 0.077],
                                     [0, 0, 1, 0.2665],
                                     [0, 0, 0, 1]])

        M_cam0433_to_A = extrinsics['043322070878'][0]
        M_cam7506_to_A = extrinsics['750612070851'][0]

        M_cam7506_to_base = M_cam7506_to_A @ np.linalg.inv(M_cam0433_to_A) @ M_cam0433_to_end @ M_end_to_base
        camera_matrix = intrinsics['750612070851']

        return M_cam7506_to_base

    def create_transformation_matrix(self, position, quaternion):
        R = self.quaternion_to_rotation_matrix(quaternion)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    def quaternion_to_rotation_matrix(self, quaternion):
        qw, qx, qy, qz = quaternion
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

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

                if angle < np.radians(75):
                    selected_points.append(trajectory_points[i])
                    selected_indices.append(i)
                    current_index = i
                    break

        return selected_indices



    def visualize_trajectory(self, trajectory_points=None, marks=None):
        projected_points = self.translate_points(trajectory_points)
        plt.clf()
        if marks is not None:
            waypoint_points = projected_points[marks]
            plt.plot(waypoint_points[:, 0], waypoint_points[:, 1], color='blue', linewidth=4)

        plt.plot(projected_points[:, 0], projected_points[:, 1], color='red', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('off')
        plt.xlim(-0.45, 0.45)
        plt.ylim(-0.5, 0.5)
        plt.gcf().set_size_inches(480/96, 480/96)
        plt.tight_layout()
        plt.savefig(f'./trajectory_visualization.png', dpi=96)

    def optimize_trajectory(self, small_demo, demo_idx, small_demo_idx, three_dimension=False):
        trajectory_points = small_demo['trajectory_points']
        frame_start = small_demo['frame_start']

        small_demo_marks = self.calculate_new_points(np.array(trajectory_points), three_dimension)
        demo_marks = [mark + frame_start for mark in small_demo_marks]

        return demo_marks

