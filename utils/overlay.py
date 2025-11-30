import cv2
import numpy as np
from utils.mask_display import mask_map
from utils.contour_detector import getting_coordinates


def painter_borders(image: np.ndarray, mask_unique: np.ndarray):
    im_overlay = image.copy()
    for mask in mask_map(mask_unique):
        for box in getting_coordinates(mask):
            print(box)
            (x, y, w, h) = [v for v in box]
            cv2.rectangle(im_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return im_overlay