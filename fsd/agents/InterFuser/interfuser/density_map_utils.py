"""
Generate density map used in InterFuser.

The density map is generated with a dimension of (R, R, 7) as follows:
    - R: range of the density map with ego vehicle as the bottom center
    - 7: density map with 7 channels, including:
        - 0: probability of the point being occupied
        - 1: distance in x-axis to the nearest instance in the scene
        - 2: distance in y-axis to the nearest instance in the scene
        - 3: yaw angle of the nearest instance in the scene
        - 4: box in x-axis of the nearest instance in the scene
        - 5: box in y-axis of the nearest instance in the scene
        - 6: velocity of the nearest instance in the scene
"""
import torch
import numpy as np 

def grid_to_xy(i, j, bev_range, pixels_per_meter):
    """Convert grid index to Lidar xy coordinates.

    Args:
        i (int): Grid index in x-axis. x is front
        j (int): Grid index in y-axis. y is left

    Returns:
        tuple: xy coordinates.
    """
    xmin, xmax, ymin, ymax = bev_range
    x = (i + 0.5)/ pixels_per_meter + xmin
    y = (j + 0.5)/ pixels_per_meter + ymin
    return x, y

def generate_density_map(gt_instances, bev_range, pixels_per_meter = 1):
    """Generate density map used in InterFuser.

    Args:
        bev_range (list[float]): BEV range of the density map.

    Returns:
        np.ndarray: Density map with shape (R, R, 7).
    """
    bev_range = [0, 20, -10, 10] # (x forward, y left)
    map = np.zeros((int((bev_range[1] - bev_range[0]) * pixels_per_meter),
                    int((bev_range[3] - bev_range[2]) * pixels_per_meter), 7),
    dtype=np.float32)
    
    # filter instances out of the bev range
    bboxes = gt_instances['gt_bboxes_3d']
    assert bboxes.box_dim >= 7, "Lidar box should have velocity augumented so that the dimension is at least 7"
    instances_velocities = bboxes.tensor[:, 6:8] # (N, 2)
    filter = bboxes.in_range_bev([bev_range[0], bev_range[2], bev_range[1], bev_range[3]])
    bboxes = bboxes[filter]
    if len(bboxes) == 0:
        return map
    
    # fill map
    center_xy_boxes = bboxes.center[:, 0:2] # (N, 2)
    
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            x, y = grid_to_xy(i, j, bev_range, pixels_per_meter)
            
            # find cloest instance
            dist = np.linalg.norm(center_xy_boxes - np.array([x, y]).reshape(-1, 2), axis=1)
            min_dist_idx = np.argmin(dist)
            min_dist = dist[min_dist_idx]
            box = bboxes[int(min_dist_idx)]
            velocity = np.linalg.norm(instances_velocities[min_dist_idx])
            prob = np.power(0.5 / max(0.5, np.sqrt(min_dist)), 0.5)
            
            map[i, j] = np.array([prob,
                                  box.center[0][0] - x,
                                  box.center[0][1] - y,
                                  box.yaw[0],
                                  box.dims[0][0],
                                  box.dims[0][1],
                                  velocity]
                        )
    map = torch.from_numpy(map)
    
    return map