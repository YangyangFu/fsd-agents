from typing import Optional
import numpy as np 
import laspy
from mmengine import check_file_exist
from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.structures.bbox_3d import get_box_type

# Crop img with a given center and size, then paste the cropped
def center_crop(image, center, size):
    """Crop image with a given center and size, then paste the cropped
    image to a blank image with two centers align.

    This function is equivalent to generating a blank image with ``size``
    as its shape. Then cover it on the original image with two centers (
    the center of blank image and the center of original image)
    aligned. The overlap area is paste from the original image and the
    outside area is filled with ``0``.

    Args:
        image (np array, H x W x C): Original image.
        center (list[int]): Target crop center coord.
        size (list[int]): Target crop size. [target_h, target_w]

    Returns:
        cropped_img (np array, target_h x target_w x C): Cropped image.
        border (np array, 4): The distance of four border of
            ``cropped_img`` to the original image area, [top, bottom,
            left, right]
        patch (list[int]): The cropped area, [left, top, right, bottom].
    """
    center_y, center_x = center
    target_h, target_w = size
    img_h, img_w, img_c = image.shape

    x0 = max(0, center_x - target_w // 2)
    x1 = min(center_x + target_w // 2, img_w)
    y0 = max(0, center_y - target_h // 2)
    y1 = min(center_y + target_h // 2, img_h)
    patch = np.array((int(x0), int(y0), int(x1), int(y1)))

    left, right = center_x - x0, x1 - center_x
    top, bottom = center_y - y0, y1 - center_y

    cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
    cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
    
    y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
    x_slice = slice(cropped_center_x - left, cropped_center_x + right)
    cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
        cropped_center_y - top, cropped_center_y + bottom,
        cropped_center_x - left, cropped_center_x + right
    ],
                        dtype=np.float32)

    return cropped_img, border, patch

# load point cloud data
def load_points_carla(
        lidar_path: str, 
        input_meta: dict, 
        coord_type: Optional[str] = 'LiDAR', 
        num_features: Optional[int]=3, 
        to_float32: Optional[bool]=True
    ) -> BasePoints:
    
    """Load point cloud data from CARLA dataset.
    
    Point cloud data in CARLA dataset is saved in ego frame in a left-hand world.
    This function loads the data and convert it to right-hand MMDET3D lidar coord

    Args:
        lidar_path (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    
    assert lidar_path.endswith('.laz'), "Only support loading laz file for now"
    assert num_features <= 4, "Only support loading xyz and intensity in laz for now"
    assert 'lidar2ego' in input_meta, "lidar2ego should be provided in input_meta"
    
    check_file_exist(lidar_path)
    
    with open(lidar_path, "rb") as f:
        las = laspy.read(f)
        if num_features == 4:
            points = np.vstack([las.x, las.y, las.z, las.intensity]).T
        elif num_features == 3:
            points = np.vstack([las.x, las.y, las.z]).T
        else: 
            raise NotImplementedError

    if to_float32:
        points = points.astype(np.float32)
    
    # convert transform
    left2right = np.eye(4)
    left2right[1, 1] = -1
    
    # mmdet lidar coord to mmdet ego coord
    lidar2ego = input_meta['lidar2ego']
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    
    # convert to mmdet3d lidar coord: ego2lidar_mmdet @ lefthand_ego2mmdet_ego
    points = (np.linalg.inv(lidar2ego) @ left2right @ points_hom.T).T 
    points = points[:, :3]
    
    # data are converted into lidar coord
    points_class = get_points_type(points_type='LiDAR') 
    _, target_mode = get_box_type(coord_type)
    
    points = points_class(
            points, points_dim=points.shape[-1]
        ).convert_to(target_mode)
    
    return points