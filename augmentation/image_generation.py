import numpy as np
import io

from PIL import Image
import matplotlib.pyplot as plt


class TrajectoryRenderer:
    def __init__(self, env, camera_name):
        self.env = env
        self.camera_name = camera_name
        self.extrinsic_matrix = env.get_camera_extrinsic_matrix(camera_name)
        print(self.extrinsic_matrix)
        self.camera_position = self.extrinsic_matrix[:3, 3]
        self.camera_rotation = self.extrinsic_matrix[:3, :3]

        # self.camera_position = np.array([1.0, 0.0, 1.75])
        # self.camera_rotation = np.array([
        #     [0.0, -0.70614724, 0.70806503],
        #     [1.0, 0.0, 0.0],
        #     [0.0, 0.70806503, 0.70614724]
        # ])

    def render_trajectory_image(self, trajectory_points, ind, samples, save_mode, rotate=False):
        frame = self.env.render(mode="rgb_array", height=480, width=480, camera_name=self.camera_name)
        image2 = Image.fromarray(frame)
        factor = self.get_save_mode_factor(save_mode)
        image_array = self.apply_image_filter(image2, factor)

        if rotate:
            trajectory_points = self.rotate_trajectory(trajectory_points, ind)

        transformed_points = np.dot(trajectory_points - self.camera_position, self.camera_rotation)

        self.save_and_append_images(transformed_points, samples, ind, image_array, rotate)


    def calculate_camera_position(self, center, distance, azim_angle):
        azim_rad = np.deg2rad(azim_angle)
        new_x = center[0] + distance * np.cos(azim_rad)
        new_y = center[1] + distance * np.sin(azim_rad)
        new_z = self.camera_position[2] 
        return np.array([new_x, new_y, new_z])
        
    def calculate_camera_rotation(self, camera_position, target_position):
        direction = target_position - camera_position
        direction = direction / np.linalg.norm(direction)  # normalization

        up = np.array([0, 1, 0])  

        right = np.cross(up, direction)
        right = right / np.linalg.norm(right)

        up = np.cross(direction, right)
        up = up / np.linalg.norm(up)

        rotation_matrix = np.vstack([right, up, -direction])

        return rotation_matrix

    def render_trajectory_lowdim(self, trajectory_points, ind, samples, save_mode, num_images):

        center = np.mean(trajectory_points, axis=0) 

        distance_to_center = np.linalg.norm(self.camera_position - center)

        for i, azim in enumerate(np.linspace(0, 360, num_images, endpoint=False)):
            new_camera_position = self.calculate_camera_position(center, distance_to_center, azim)
            new_camera_rotation = self.calculate_camera_rotation(new_camera_position, center)
            prev_camera_position = self.camera_position
            prev_camera_rotation = self.camera_rotation

            self.camera_position = new_camera_position
            self.camera_rotation = new_camera_rotation
            
            self.render_trajectory_image(trajectory_points, ind + i, samples, save_mode, rotate=False)

            self.camera_position = prev_camera_position
            self.camera_rotation = prev_camera_rotation



    def get_save_mode_factor(self, save_mode):
        if save_mode == 'lowdim':
            return 0
        elif save_mode == 'image':
            return 0.5
        elif save_mode == 'realword':
            raise ValueError("The 'realword' mode is not yet implemented.")
        else:
            raise ValueError(f"Unknown save_mode: {save_mode}")

    def apply_image_filter(self, image, factor):
        image_array = np.array(image)
        white_image = np.ones_like(image_array) * 255
        new_image_array = (image_array * factor + white_image * (1 - factor)).astype(np.uint8)
        return Image.fromarray(new_image_array)

    def rotate_trajectory(self, trajectory_points, i):
        start_point, end_point, middle_points = trajectory_points[0], trajectory_points[-1], trajectory_points[1:-1]
        axis = (end_point - start_point) / np.linalg.norm(end_point - start_point)
        angle = np.deg2rad(i * 360 / 30)  # 通过角度旋转产生多个图像
        rotation_matrix = self.get_rotation_matrix(axis, angle)

        rotated_middle_points = np.dot(middle_points - start_point, rotation_matrix.T) + start_point
        return np.vstack([start_point, rotated_middle_points, end_point])

    def get_rotation_matrix(self, axis, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return np.array([
            [cos_angle + axis[0]**2 * (1 - cos_angle), axis[0]*axis[1]*(1 - cos_angle) - axis[2]*sin_angle, axis[0]*axis[2]*(1 - cos_angle) + axis[1]*sin_angle],
            [axis[1]*axis[0]*(1 - cos_angle) + axis[2]*sin_angle, cos_angle + axis[1]**2 * (1 - cos_angle), axis[1]*axis[2]*(1 - cos_angle) - axis[0]*sin_angle],
            [axis[2]*axis[0]*(1 - cos_angle) - axis[1]*sin_angle, axis[2]*axis[1]*(1 - cos_angle) + axis[0]*sin_angle, cos_angle + axis[2]**2 * (1 - cos_angle)]
        ])

    def save_and_append_images(self, transformed_points, samples, ind, image_array, rotate):

        plt.clf()
        projected_points = transformed_points[:, :2]
        plt.plot(projected_points[:, 0], -projected_points[:, 1], color='red', linewidth=5)
        plt.axis('off')
        plt.xlim(-0.45, 0.45)
        plt.ylim(-0.5, 0.5)
        plt.gcf().set_size_inches(480/96, 480/96)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=96)
        buf.seek(0)

        image1 = Image.open(buf)
        image_array.paste(image1, (0, 0), image1)
        image_array.save(f'./tmp/image_com_{ind}.png')

        samples['images'].append(np.array(image_array))
        samples['labels'].append(1 if rotate else 0)
        buf.close()


class TrajectoryNoiseGenerator:
    def __init__(self, trajectory_points):
        self.trajectory_points = trajectory_points

    def render_one_point(self):
        start_index = np.random.randint(0, len(self.trajectory_points) - 10)
        end_index = start_index + 1

        sub_trajectory = self.trajectory_points[start_index:end_index]
        noisy_sub_trajectory = self.add_noise(sub_trajectory, scale_range=(0.02, 0.05))

        noise_trajectory = self.trajectory_points.copy()
        noise_trajectory[start_index:end_index] = noisy_sub_trajectory
        self.remove_adjacent_points(noise_trajectory, start_index)
        return noise_trajectory

    def add_noise(self, sub_trajectory, scale_range):
        noise_scale = np.random.uniform(*scale_range)
        noise = noise_scale * np.random.randn(sub_trajectory.shape[0], sub_trajectory.shape[1])
        return sub_trajectory + noise

    def remove_adjacent_points(self, noise_trajectory, start_index):
        if start_index > 1:
            for i in range(1, 4):
                noise_trajectory = np.delete(noise_trajectory, start_index - i, axis=0)

    def render_one_point_circle(self):
        start_index = np.random.randint(0, min(len(self.trajectory_points) - 10, len(self.trajectory_points)))
        end_index = start_index + np.random.randint(5, 10)

        sub_trajectory = self.trajectory_points[start_index:end_index]
        noisy_sub_trajectory = self.add_noise(sub_trajectory, scale_range=(0.02, 0.04))

        inserted_indices = self.get_inserted_indices(start_index, end_index, num_points_range=(10, 20))
        for i, index in enumerate(inserted_indices):
            self.trajectory_points = np.insert(self.trajectory_points, index + i, sub_trajectory[0] + np.random.uniform(0.04, 0.06)*np.random.randn(), axis=0)
        return self.trajectory_points

    def get_inserted_indices(self, start_index, end_index, num_points_range):
        num_inserted_points = np.random.randint(*num_points_range)
        inserted_indices = np.random.randint(start_index, end_index, size=num_inserted_points)
        inserted_indices.sort()
        return inserted_indices

    def render_series(self):
        start_index = np.random.randint(0, max(1, len(self.trajectory_points) - 20))
        end_index = start_index + np.random.randint(5, 10)
        sub_trajectory = self.trajectory_points[start_index:end_index]

        noisy_sub_trajectory = self.add_noise(sub_trajectory, scale_range=(0.03, 0.06))
        noise_trajectory = self.trajectory_points.copy()
        noise_trajectory[start_index:end_index] = noisy_sub_trajectory
        return noise_trajectory


class TrajectoryGenerator:
    def __init__(self, trajectory_points):
        self.trajectory_points = trajectory_points
        self.noise_generator = TrajectoryNoiseGenerator(trajectory_points)

    def generate_negative_trajectory_points(self):
        num_one_point = np.random.randint(0, 8)
        noise_trajectory = self.trajectory_points.copy()

        for _ in range(num_one_point):
            noise_trajectory = self.noise_generator.render_one_point()

        if num_one_point == 0:
            one_point_circle_flag = 1
        else:
            one_point_circle_flag = np.random.randint(2)

        if one_point_circle_flag:
            noise_trajectory = self.noise_generator.render_one_point_circle()

        if not one_point_circle_flag and np.random.randint(2):
            noise_trajectory = self.noise_generator.render_series()

        return noise_trajectory


class PictureGenerator:
    def __init__(self, renderer, samples, save_mode):
        self.renderer = renderer
        self.samples = samples
        self.save_mode = save_mode

    def generate_positive_picture(self, trajectory_points, ind, num_images=30):
        for i in range(num_images):
            self.renderer.render_trajectory_image(trajectory_points, i, self.samples, self.save_mode, rotate=True)
        if self.save_mode == "image":
            pass
        elif self.save_mode == "lowdim":
            pass
            # This part can be expanded if needed

    def generate_negative_picture(self, trajectory_points, ind, num_images):
        for i in range(num_images):
            noise_trajectory = TrajectoryGenerator(trajectory_points).generate_negative_trajectory_points()
            self.renderer.render_trajectory_image(noise_trajectory, i, self.samples, self.save_mode, rotate=False)


