import cv2
import numpy as np
from PIL import Image

from images_eval import (
    compute_boundary_f1,
    compute_contour_accuracy,
    compute_miou,
    compute_overall_contour_accuracy,
)


def mask_segmentation(img) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors, inverse = np.unique(img_rgb.reshape(-1, 3), axis=0, return_inverse=True)
    mask_indices = inverse.reshape(img_rgb.shape[:2])

    print("Классы:", np.unique(mask_indices))
    return mask_indices


def main():
    mask = np.array(Image.open("images/00000.png"))
    print(np.unique(mask))
    gt_mask = np.array(cv2.imread("images/00000.png"))
    print(np.unique(gt_mask))

    img = np.array(cv2.imread("images/00000.png"))  # BGR
    mask_segmentation(img)

    gt_mask = mask_segmentation(np.array(cv2.imread("images/cats_rac_test.png")))
    pred_mask = mask_segmentation(np.array(cv2.imread("images/cats_rac_sam.png")))

    print("Уникальные значения в test:", np.unique(gt_mask))
    print("Уникальные значения в sam:", np.unique(pred_mask))

    miou = compute_miou(pred_mask, gt_mask)
    bf = compute_boundary_f1(pred_mask, gt_mask)
    hausdorff = compute_contour_accuracy(pred_mask, gt_mask, metric="hausdorff")
    chamfer = compute_contour_accuracy(pred_mask, gt_mask, metric="chamfer")
    print(f"mIoU: {miou:.4f}")
    print(f"Boundary F1: {bf:.4f}")
    print(f"Hausdorff distance: {hausdorff:.2f}")
    print(f"Chamfer distance: {chamfer:.2f}")


if __name__ == "__main__":
    main()
