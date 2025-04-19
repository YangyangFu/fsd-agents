""" This script visualizes the yaw of 3D bounding boxes in the nuScenes dataset.
    It shows the yaw of the boxes in the LiDAR frame, which is defined as the
    rotation around the Z-axis in the right-handed coordinate system. The yaw=0 at positive X-axis
    and yaw=90 at positive Y-axis. The script filters the boxes based on their yaw values
    and renders them in black if they are within a specified range. 
"""
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes


min_yaw_deg = 0  # Tinker around with this value
max_yaw_deg = 90  # Tinker around with this value


# Load up the nuScenes mini split.
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes-v1.0/mini', verbose=False)


# Select the 10th sample.
sample = nusc.sample[10]
sample_data_token = sample['data']['LIDAR_TOP']

# Get the boxes belonging to the sample in the ego frame.
#_, boxes, _ = nusc.get_sample_data(sample_data_token, use_flat_vehicle_coordinates=True)

# get box to lidar frame
_, boxes, _ = nusc.get_sample_data(sample_data_token)

# Filter for only boxes which belong to vehicles (these boxes are bigger compared to other classes like pedestrians and 
# cones, which makes it easier to observe them)
boxes = [bb for bb in boxes if 'vehicle' in bb.name]

_, ax = plt.subplots(1, 1, figsize=(9, 9))

# Go through each box.
for box in boxes:
    # For boxes which have a yaw within the desired range, render them in black.
    if min_yaw_deg < np.rad2deg(Quaternion(box.orientation).yaw_pitch_roll[0]) < max_yaw_deg:
        c = [0, 0, 0]
    else:
        c = np.array(nusc.explorer.get_color(box.name)) / 255.0
    box.render(ax, view=np.eye(4), colors=(c, c, c))

axes_limit = 40
ax.set_xlim(-axes_limit, axes_limit)
ax.set_ylim(-axes_limit, axes_limit)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()

