import cv2
import numpy as np
from images_eval import (
    compute_miou,
    compute_boundary_f1,
    compute_contour_accuracy,
    compute_overall_contour_accuracy,
)


def main():
    gt_mask = cv2.imread('images/cats_rac_test.png')
    pred_mask = cv2.imread('images/cats_rac_sam.png')

    print("Уникальные значения в test:", np.unique(gt_mask))
    print("Уникальные значения в sam:", np.unique(pred_mask))

    miou = compute_miou(pred_mask, gt_mask)
    bf = compute_boundary_f1(pred_mask, gt_mask)
    hausdorff = compute_contour_accuracy(pred_mask, gt_mask, metric='hausdorff')
    chamfer = compute_contour_accuracy(pred_mask, gt_mask, metric='chamfer')
    print(f"mIoU: {miou:.4f}")
    print(f"Boundary F1: {bf:.4f}")
    print(f"Hausdorff distance: {hausdorff:.2f}")
    print(f"Chamfer distance: {chamfer:.2f}")

if __name__ == "__main__":
    main()
