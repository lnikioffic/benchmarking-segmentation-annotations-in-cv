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

# !!! Импорт метрик для рамок
from frames_eval import (
    get_box_for_id,
    BoxEvaluator,
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
    visualize_wb_mask,
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

    # print("Классы:", np.unique(mask_indices))
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
    dataset = read_dataset("train")

    seg_j = []
    seg_f = []
    seg_h = []
    seg_c = []
    video_j = []  # IoU scores
    video_f = []  # Boundary scores
    video_h = []  # Hausdorff scores
    video_c = []  # Chamfer scores

    # Инициализация Evaluator для рамок
    box_evaluator = BoxEvaluator(iou_threshold=0.5)

    for folder in dataset:
        image_path, mask_path = folder[0]
        print(image_path)
        mask = np.array(cv2.imread(mask_path))
        coords = get_coordinates(mask)
        image = np.array(cv2.imread(image_path))
        segmentation_mask = segmentation(image, coords)
        assert len(np.unique(segmentation_mask)) == len(
            np.unique(mask_segmentation(mask))
        )
        # mask_view = visualize_unique_mask(segmentation_mask)
        # cv2.imshow("mask1", mask_view)
        miou = compute_miou(segmentation_mask, mask)
        bf = compute_boundary_f1(segmentation_mask, mask)
        hausdorff = compute_contour_accuracy(
            segmentation_mask, mask, metric="hausdorff"
        )
        chamfer = compute_contour_accuracy(segmentation_mask, mask, metric="chamfer")
        seg_j.append(miou)
        seg_f.append(bf)
        seg_h.append(hausdorff)
        seg_c.append(chamfer)
        print(f"mIoU: {miou:.4f}")
        print(f"Boundary F1: {bf:.4f}")
        print(f"Hausdorff distance: {hausdorff:.2f}")
        print(f"Chamfer distance: {chamfer:.2f}")
        masks = tracker(folder, segmentation_mask)

        for mask_t, data in zip(masks, folder):
            cv2.imshow("mask", visualize_wb_mask(mask_t))
            cv2.waitKey(1)

            gt_img_raw = np.array(cv2.imread(data[1]))
            gt_mask = mask_segmentation(gt_img_raw)

            # 1. Метрики Масок
            video_j.append(compute_miou(mask_t, gt_mask))
            video_f.append(compute_boundary_f1(mask_t, gt_mask))
            video_h.append(
                compute_contour_accuracy(mask_t, gt_mask, metric="hausdorff")
            )
            video_c.append(compute_contour_accuracy(mask_t, gt_mask, metric="chamfer"))

            # 2. Метрики Рамок (Сравнение по ID объектов)
            gt_ids = np.unique(gt_mask)
            gt_ids = gt_ids[gt_ids != 0]  # Исключаем фон

            pred_ids = np.unique(mask_t)
            pred_ids = pred_ids[pred_ids != 0]  # Исключаем фон

            all_ids = set(gt_ids) | set(pred_ids)

            for obj_id in all_ids:
                gt_box = get_box_for_id(gt_mask, obj_id)
                pred_box = get_box_for_id(mask_t, obj_id)

                # Обновляем статистику через Evaluator
                box_evaluator.update(pred_box, gt_box)

    # --- Подсчет финальных метрик рамок ---
    box_res = box_evaluator.compute()

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    mean_j = np.mean(video_j)
    mean_f = np.mean(video_f)

    print("\n" + "=" * 60)
    print(f"{'BENCHMARK RESULTS':^60}")
    print("=" * 60)

    print(f"\n{'- TRACKING: MASKS -':^60}")
    print(f"  Mean mIoU (J)        : {mean_j:.4f}")
    print(f"  Mean Boundary F1 (F) : {mean_f:.4f}")
    print(f"  Global Score (J&F)   : {((mean_j + mean_f)/2):.4f}")
    print(f"  Mean Hausdorff       : {np.mean(video_h):.2f}")
    print(f"  Mean Chamfer         : {np.mean(video_c):.2f}")

    print(f"\n{'- TRACKING: BOUNDING BOXES -':^60}")
    if box_res["mean_iou"] > 0:
        print(f"  Mean Box IoU         : {box_res['mean_iou']:.4f}")
        print(f"  Mean Center Error    : {box_res['mean_dist']:.2f} px")
        print(f"  Box F1 Score (@0.5)  : {box_res['f1']:.4f}")
        print(f"    * Precision        : {box_res['precision']:.4f}")
        print(f"    * Recall           : {box_res['recall']:.4f}")
    else:
        print("  No bounding boxes detected.")

    print(f"\n{'- SEGMENTATION (INIT) -':^60}")
    print(f"  Mean mIoU            : {np.mean(seg_j):.4f}")
    print(f"  Mean Boundary F1     : {np.mean(seg_f):.4f}")
    print(f"  Mean Hausdorff       : {np.mean(seg_h):.2f}")
    print(f"  Mean Chamfer         : {np.mean(seg_c):.2f}")
    print("=" * 60 + "\n")

    # print("-" * 100)
    # try:
    #     i = np.array(cv2.imread("00000.png"))
    #     if i is not None:
    #         im = mask_segmentation(i)
    #         gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    #         get_coordinates(i)
    #         print(len(mask_map(im)))
    #         im = painter_borders(i, gray)
    #         cv2.imshow("imgg", im)

    #     img_mask = np.array(cv2.imread("images/00000.png"))
    #     if img_mask is not None:
    #         mask_segmentation(img_mask)
    #         gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    #         print(get_filtered_bboxes(gray, min_area_ratio=0.001))
    #         box = get_filtered_bboxes(gray, min_area_ratio=0.001)
    #         print("box", box)
    # except Exception:
    #     pass

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

    # try:
    #     mask = np.array(cv2.imread("images/cats_rac_test.png"))
    #     if mask is not None:
    #         print("Кошки ориг")
    #         get_coordinates(mask)
    #         c_mask = np.array(cv2.imread("images/cats_rac_sam.png"))
    #         print("Кошки сам")
    #         get_coordinates(c_mask)
    #         print("----------")
    #         gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #         for m in mask_map(gray):
    #             print(getting_coordinates(m))
    #         im = painter_borders(mask, gray)
    #         cv2.imshow("img", im)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    #         gt_mask = mask_segmentation(
    #             np.array(cv2.imread("images/cats_rac_test.png"))
    #         )
    #         pred_mask = mask_segmentation(
    #             np.array(cv2.imread("images/cats_rac_sam.png"))
    #         )

    #         print("Уникальные значения в test:", np.unique(gt_mask))
    #         print("Уникальные значения в sam:", np.unique(pred_mask))

    #         miou = compute_miou(pred_mask, gt_mask)
    #         bf = compute_boundary_f1(pred_mask, gt_mask)
    #         hausdorff = compute_contour_accuracy(pred_mask, gt_mask, metric="hausdorff")
    #         chamfer = compute_contour_accuracy(pred_mask, gt_mask, metric="chamfer")
    #         print(f"mIoU: {miou:.4f}")
    #         print(f"Boundary F1: {bf:.4f}")
    #         print(f"Hausdorff distance: {hausdorff:.2f}")
    #         print(f"Chamfer distance: {chamfer:.2f}")
    # except Exception:
    #     pass


if __name__ == "__main__":
    main()
