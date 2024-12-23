

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TrajectoryPoint:
    """Represents a point in the trajectory with its coordinates."""
    coordinates: np.ndarray
    index: int

class GeometryCalculator:
    
    @staticmethod
    def calculate_vector(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
        return point2 - point1

    @staticmethod
    def calculate_angle(vector1: np.ndarray, vector2: np.ndarray) -> float:
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        cos_theta = dot_product / (norm_vector1 * norm_vector2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

class CoordinateTransformer:
    """Handles coordinate transformations between different reference frames."""
    def __init__(self, env, real_world: bool = False, calib_dir: Optional[str] = None):
        self.env = env
        self.real_world = real_world
        self.calib_dir = calib_dir

    def transform_points(self, trajectory_points: np.ndarray) -> np.ndarray:
        if self.real_world:
            transformed_points = self._transform_real_world(trajectory_points)
        else:
            transformed_points = self._transform_simulation(trajectory_points)
        
        return transformed_points[:, :2]  # Return only x,y coordinates

    def _transform_real_world(self, points: np.ndarray) -> np.ndarray:
        from utils.realworld_utils import translate_points
        return translate_points(self.calib_dir, points)

    def _transform_simulation(self, points: np.ndarray) -> np.ndarray:
        extrinsic_matrix = self.env.get_camera_extrinsic_matrix('agentview')
        camera_position = extrinsic_matrix[:3, 3]
        camera_rotation = extrinsic_matrix[:3, :3]
        return np.dot(points - camera_position, camera_rotation)


class TrajectoryOptimizer:
    """Main class for optimizing robot trajectories."""
    def __init__(self, env, real_world: bool = False, calib_dir: Optional[str] = None):
        self.geometry = GeometryCalculator()
        self.transformer = CoordinateTransformer(env, real_world, calib_dir)

    def optimize_trajectory(self, demo: Dict, demo_idx: int, small_demo_idx: int, 
                          three_dimension: bool = False) -> List[int]:
        frame_start = demo['frame_start']
        waypoints = self._calculate_waypoints_dp(demo, three_dimension)
        return [mark + frame_start for mark in waypoints]

    def _calculate_waypoints_dp(self, demo: Dict, three_dimension: bool) -> List[int]:
        err_threshold = 0.005
        actions = demo['actions']
        gt_states = demo['gt_states']
        num_frames = len(actions)
        
        dp_table = self._initialize_dp_table(num_frames)
        
        min_error = self._compute_trajectory_errors(actions, gt_states, list(range(1, num_frames)))
        if err_threshold < min_error:
            return list(range(1, num_frames))

        return self._fill_dp_table(dp_table, actions, gt_states, num_frames, err_threshold)

    def _initialize_dp_table(self, size: int) -> Dict:
        dp_table = {i: (0, []) for i in range(size)}
        dp_table[1] = (1, [1])
        return dp_table

    def _compute_trajectory_errors(self, actions: np.ndarray, gt_states: np.ndarray, 
                                 waypoints: List[int]) -> float:
        from utils.trajectory_utils import compute_errors
        return compute_errors(actions=actions, gt_states=gt_states, waypoints=waypoints)

    def _fill_dp_table(self, dp_table: Dict, actions: np.ndarray, gt_states: np.ndarray, 
                      num_frames: int, err_threshold: float) -> List[int]:
        initial_waypoints = [0, num_frames - 1]

        for i in range(1, num_frames):
            min_waypoints_required = float("inf")
            best_waypoints = []

            for k in range(1, i):
                waypoints = [j - k for j in initial_waypoints if k <= j < i] + [i - k]
                total_err = self._compute_trajectory_errors(
                    actions[k:i + 1], gt_states[k:i + 1], waypoints
                )

                if total_err < err_threshold:
                    prev_count, prev_waypoints = dp_table[k - 1]
                    total_count = 1 + prev_count

                    if total_count < min_waypoints_required:
                        min_waypoints_required = total_count
                        best_waypoints = prev_waypoints + [i]

            dp_table[i] = (min_waypoints_required, best_waypoints)

        _, waypoints = dp_table[num_frames - 1]
        waypoints.extend(initial_waypoints)
        return sorted(list(set(waypoints)))

    def _calculate_geometric_waypoints(self, demo: Dict, three_dimension: bool) -> List[int]:
        points = np.array(demo['trajectory_points'])
        trajectory_points = points if three_dimension else self.transformer.transform_points(points)
        
        tolerance = self._calculate_tolerance(trajectory_points)
        
        return self._select_waypoints(trajectory_points, tolerance)

    def _calculate_tolerance(self, points: np.ndarray) -> float:
        return np.max(np.linalg.norm(points[1:] - points[:-1], axis=1)) * 2

    def _select_waypoints(self, points: np.ndarray, tolerance: float) -> List[int]:
        selected_indices = [0]
        current_idx = 0

        while current_idx < len(points) - 1:
            if np.array_equal(points[current_idx], points[-1]):
                break

            next_idx = self._find_next_waypoint(
                points, current_idx, selected_indices, tolerance
            )
            
            if next_idx is None:
                break
                
            selected_indices.append(next_idx)
            current_idx = next_idx

        return selected_indices

    def _find_next_waypoint(self, points: np.ndarray, current_idx: int, 
                           selected_indices: List[int], tolerance: float) -> Optional[int]:
        candidates = []
        
        for i in range(len(points)):
            if i in selected_indices:
                continue
                
            distance = np.linalg.norm(points[current_idx] - points[i])
            if distance > tolerance:
                continue
                
            vector_to_candidate = self.geometry.calculate_vector(points[current_idx], points[i])
            vector_to_goal = self.geometry.calculate_vector(points[current_idx], points[-1])
            angle = self.geometry.calculate_angle(vector_to_candidate, vector_to_goal)
            
            if angle < np.radians(DELTA_THETA):
                candidates.append((angle, distance, i))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]