import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from images_eval import (
    compute_boundary_f1,
    compute_contour_accuracy,
    compute_miou,
    compute_overall_contour_accuracy,
)
from segmenter import Segmenter
from tracker_core_xmem2 import TrackerCore
from utils.contour_detector import (
    get_filtered_bboxes,
    get_filtered_bboxes_xywh,
    getting_coordinates,
)
from utils.converter import extract_color_regions, merge_masks
from utils.mask_display import (  # потестить mask_map на масках с несколькими объектами и посмотреть порядок потом как сам их показывает должен быть одинаковый и тогда можно смореть с несколькими объектами
    mask_map,
    visualize_unique_mask,
)
from utils.overlay import painter_borders
from XMem2.inference.interact.interactive_utils import overlay_davis

BASE_PATH = "DAVIS"
IMAGES_PATH = os.path.join(BASE_PATH, "JPEGImages", "480p")
MASKS_PATH = os.path.join(BASE_PATH, "Annotations", "480p")


# Таблиаца для оценки с начало идёт оценка входной маски, потом идёт оценка по трекеру


def mask_segmentation(img) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors, inverse = np.unique(img_rgb.reshape(-1, 3), axis=0, return_inverse=True)
    mask_indices = inverse.reshape(img_rgb.shape[:2])

    print("Классы:", np.unique(mask_indices))
    return mask_indices


def read_dataset(select_name=None):
    if select_name is not None:
        images_folder = filter(lambda x: select_name in x, os.listdir(IMAGES_PATH))
        mask_folder = filter(lambda x: select_name in x, os.listdir(MASKS_PATH))
    else:
        images_folder = os.listdir(IMAGES_PATH)
        mask_folder = os.listdir(MASKS_PATH)

    data = []
    for images_path, mask_path in zip(images_folder, mask_folder):
        images_path = os.path.join(IMAGES_PATH, images_path)
        mask_path = os.path.join(MASKS_PATH, mask_path)
        segment = []
        for image, mask in zip(os.listdir(images_path), os.listdir(mask_path)):
            image = os.path.join(images_path, image)
            mask = os.path.join(mask_path, mask)
            segment.append((image, mask))
        data.append(segment)

    return data


def get_coordinates(org_mask):
    seg_mask = mask_segmentation(org_mask)
    map = mask_map(seg_mask)
    coords = []
    for obj in map:
        m = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        coords.extend(get_filtered_bboxes(m, min_area_ratio=0.001))

    print(coords)
    return coords


def segmentation(image, coords):
    segmenter = Segmenter()
    segmenter.set_image(image)
    mask_objects = []
    for coord in coords:
        prompt = {"boxes": np.array([coord])}
        masks, scores, logits = segmenter.predict(prompt, "box", True)
        mask_objects.append(masks[np.argmax(scores)])

    mask, unique_mask = merge_masks(mask_objects)
    mask_indices = mask_segmentation(unique_mask)
    return mask_indices


def tracker(folder: list, mask):
    tracker_core = TrackerCore()
    masks = []
    for i in tqdm(range(len(folder)), desc="Tracking"):
        image = np.array(cv2.imread(folder[i][0]))
        if i == 0:
            mask = tracker_core.track(
                image,
                mask,
            )
            masks.append(mask)
        else:
            mask = tracker_core.track(image)
            masks.append(mask)
    return masks


def main():
    dataset = read_dataset("bear")
    for folder in dataset:
        image_path, mask_path = folder[0]
        mask = np.array(cv2.imread(mask_path))
        coords = get_coordinates(mask)
        image = np.array(cv2.imread(image_path))
        segmentation_mask = segmentation(image, coords)
        assert len(np.unique(segmentation_mask)) == len(
            np.unique(mask_segmentation(mask))
        )

        miou = compute_miou(segmentation_mask, mask)
        bf = compute_boundary_f1(segmentation_mask, mask)
        hausdorff = compute_contour_accuracy(
            segmentation_mask, mask, metric="hausdorff"
        )
        chamfer = compute_contour_accuracy(segmentation_mask, mask, metric="chamfer")
        print(f"mIoU: {miou:.4f}")
        print(f"Boundary F1: {bf:.4f}")
        print(f"Hausdorff distance: {hausdorff:.2f}")
        print(f"Chamfer distance: {chamfer:.2f}")
        masks = tracker(folder, segmentation_mask)
        print(len(masks))
    # mask = np.array(Image.open("images/00000.png"))
    # print(np.unique(mask))
    # gt_mask = np.array(cv2.imread("images/00000.png"))
    # print(np.unique(gt_mask))
    print("-" * 100)
    i = np.array(cv2.imread("00000.png"))
    im = mask_segmentation(i)
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    get_coordinates(i)
    print(len(mask_map(im)))
    im = painter_borders(i, gray)
    cv2.imshow("imgg", im)

    img_mask = np.array(cv2.imread("images/00000.png"))  # BGR
    mask_segmentation(img_mask)
    gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    print(get_filtered_bboxes(gray, min_area_ratio=0.001))
    box = get_filtered_bboxes(gray, min_area_ratio=0.001)

    print("box", box)
    # image = np.array(cv2.imread("images/00000.jpg"))
    # im = painter_borders(image, gray)
    # cv2.imshow("img", im)
    # segmenter = Segmenter()
    # segmenter.set_image(image)

    # prompt = {
    #     "boxes": np.array(box),
    # }
    # maskss = []
    # masks, scores, logits = segmenter.predict(prompt, mode="box")
    # maskss.append(masks[np.argmax(scores)])

    # mask, unique_mask = merge_masks(maskss)

    # mask_indices, colors = extract_color_regions(unique_mask)
    # print("Классы:", np.unique(mask_indices))

    # f = overlay_davis(image, mask_indices)
    # f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    # mask = visualize_unique_mask(mask_indices)
    # cv2.imshow("mask", mask)
    # cv2.imshow("overlay", f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = np.array(cv2.imread("images/cats_rac_test.png"))
    print("Кошки ориг")
    get_coordinates(mask)  # есть определённый порядок
    c_mask = np.array(cv2.imread("images/cats_rac_sam.png"))
    print("Кошки сам")
    get_coordinates(c_mask)
    print("----------")
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Последним идёт самая задняя кошка
    for m in mask_map(gray):
        # print(get_filtered_bboxes_xywh(m, min_area_ratio=0.001))
        # print(get_filtered_bboxes(m, min_area_ratio=0.001))
        print(getting_coordinates(m))
    im = painter_borders(mask, gray)
    cv2.imshow("img", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
