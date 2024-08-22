import numpy as np 

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
    