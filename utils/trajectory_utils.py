import numpy as np
import wandb
from scipy.spatial.transform import Rotation
import robosuite.utils.transform_utils as T

def find_all_change_indices(data):
    changes_indices = []
    prev_value = data[0][-1]
    
    for i in range(1, len(data)):
        curr_value = data[i][-1]
        if (prev_value == -1 and curr_value == 1) or (prev_value == 1 and curr_value == -1):
            changes_indices.append(i)
        prev_value = curr_value
    
    return changes_indices

def slice_trajectory_and_states(trajectory_points, states, change_indices):
    trajectory_slices, state_slices = [], []
    start_idx = 0
    
    for idx in change_indices:
        if idx - start_idx >= 10:
            trajectory_slices.append(trajectory_points[start_idx:idx])
            state_slices.append(states[start_idx:idx])
        start_idx = idx
    
    # Handle the last slice if it's long enough
    if len(trajectory_points) - start_idx > 15:
        trajectory_slices.append(trajectory_points[start_idx:])
        state_slices.append(states[start_idx:])
    
    return trajectory_slices, state_slices

def compute_errors(actions, gt_states, waypoints, return_list=False):
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
        
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]
    keypoints_pos = [actions[k, :3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]
    
    errors = []
    
    for i in range(len(waypoints) - 1):
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        
        segment_pos = gt_pos[start_idx:end_idx]
        segment_quat = gt_quat[start_idx:end_idx]
        
        for j in range(len(segment_pos)):
            line_vector = keypoints_pos[i + 1] - keypoints_pos[i]
            point_vector = segment_pos[j] - keypoints_pos[i]
            t = np.clip(
                np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector),
                0, 1
            )
            proj_point = keypoints_pos[i] + t * line_vector
            pos_err = np.linalg.norm(segment_pos[j] - proj_point)
            
            pred_quat = T.quat_slerp(
                keypoints_quat[i], 
                keypoints_quat[i + 1], 
                fraction=j/len(segment_quat)
            )
            quat_err = (
                Rotation.from_quat(pred_quat) * 
                Rotation.from_quat(segment_quat[j]).inv()
            ).magnitude()
            
            errors.append(pos_err + quat_err)
    
    max_error = np.max(errors)
    return (max_error, errors) if return_list else max_error
