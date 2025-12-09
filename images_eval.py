import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries


def preprocess_mask(mask):
    """Привести маску к бинарному виду (0/1)."""
    if len(mask.shape) == 3:  # RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    return mask


def compute_miou(pred, gt):
    """mIoU для одной маски (Jaccard Index)."""
    pred, gt = preprocess_mask(pred), preprocess_mask(gt)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # обе маски пустые → идеальное совпадение
    return intersection / union


def compute_boundary_f1(pred, gt, tolerance=1):
    """Boundary F1 score (BF) — как в DAVIS."""
    pred, gt = preprocess_mask(pred), preprocess_mask(gt)

    # Найти границы
    pred_bound = find_boundaries(pred, mode="thick").astype(np.uint8)
    gt_bound = find_boundaries(gt, mode="thick").astype(np.uint8)

    # Расстояния от границ
    pred_dist = cv2.distanceTransform(1 - pred_bound, cv2.DIST_L2, 0)
    gt_dist = cv2.distanceTransform(1 - gt_bound, cv2.DIST_L2, 0)

    # Точки GT, близкие к pred (в пределах tolerance)
    gt_match = (pred_dist[gt_bound == 1] <= tolerance).sum()
    pred_match = (gt_dist[pred_bound == 1] <= tolerance).sum()

    n_gt = gt_bound.sum()
    n_pred = pred_bound.sum()

    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    precision = pred_match / n_pred
    recall = gt_match / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_contour_accuracy(pred, gt, metric="hausdorff"):
    """Сравнение контуров через Hausdorff или Chamfer distance."""
    pred, gt = preprocess_mask(pred), preprocess_mask(gt)

    # Получить контуры как списки координат
    def mask_to_coords(mask):
        # Убедимся, что маска имеет правильный тип данных
        mask = mask.astype(np.uint8)

        # Найти контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Если контуры найдены, объединяем их в один массив координат
        if contours:
            # Убираем лишние размерности
            contour = np.vstack(contours).squeeze()
            return contour
        else:
            # Возвращаем пустой массив, если контуров нет
            return np.empty((0, 2), dtype=np.int32)

    pred_coords = mask_to_coords(pred)
    gt_coords = mask_to_coords(gt)
    
    pred_empty = len(pred_coords) == 0
    gt_empty = len(gt_coords) == 0

    if pred_empty and gt_empty:
        # Оба контура пустые - идеальное совпадение
        return 0.0
    elif pred_empty or gt_empty:
        # Один контур пустой, другой нет - максимальная ошибка
        # Возвращаем большое число или специфичное значение
        return float('inf')
        
    if metric == "hausdorff":
        # Directed Hausdorff (берём max из двух направлений)
        d1 = directed_hausdorff(pred_coords, gt_coords)[0]
        d2 = directed_hausdorff(gt_coords, pred_coords)[0]
        return max(d1, d2)
    elif metric == "chamfer":
        # Среднее попарное расстояние (упрощённо)
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(pred_coords, gt_coords, metric="euclidean")
        chamfer = dist_matrix.min(axis=0).mean() + dist_matrix.min(axis=1).mean()
        return chamfer
    else:
        raise ValueError("metric must be 'hausdorff' or 'chamfer'")


def get_binary_masks_from_multiclass(multiclass_mask):
    """Разделить многоклассовую маску на список бинарных масок по уникальным значениям."""
    unique_vals = np.unique(multiclass_mask)
    binary_masks = []
    for val in unique_vals:
        if val == 0:  # фон — пропускаем
            continue
        binary_mask = (multiclass_mask == val).astype(np.uint8)
        binary_masks.append(binary_mask)
    return binary_masks


def compute_contour_accuracy_for_pair(pred_binary, gt_binary, metric="hausdorff"):
    """Вычислить контурную метрику для одной пары бинарных масок."""
    pred_binary, gt_binary = preprocess_mask(pred_binary), preprocess_mask(gt_binary)

    def mask_to_coords(mask):
        """Преобразует бинарную маску в контур (границу объекта) с использованием OpenCV"""

        # Убедимся, что маска имеет правильный тип данных
        mask = mask.astype(np.uint8)

        # Найти контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Если контуры найдены, объединяем их в один массив координат
        if contours:
            # Убираем лишние размерности
            contour = np.vstack(contours).squeeze()
            return contour
        else:
            # Возвращаем пустой массив, если контуров нет
            return np.empty((0, 2), dtype=np.int32)

    pred_coords = mask_to_coords(pred_binary)
    gt_coords = mask_to_coords(gt_binary)

    if metric == "hausdorff":
        d1 = directed_hausdorff(pred_coords, gt_coords)[0]
        d2 = directed_hausdorff(gt_coords, pred_coords)[0]
        return max(d1, d2)
    elif metric == "chamfer":
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(pred_coords, gt_coords, metric="euclidean")
        chamfer = dist_matrix.min(axis=0).mean() + dist_matrix.min(axis=1).mean()
        return chamfer
    else:
        raise ValueError("metric must be 'hausdorff' or 'chamfer'")


def match_objects_by_iou(pred_masks, gt_masks, iou_threshold=0.1):
    """Сопоставить объекты по максимальному IoU."""
    matches = {}
    used_gt = set()
    for i, pred_mask in enumerate(pred_masks):
        best_iou = -1
        best_j = -1
        for j, gt_mask in enumerate(gt_masks):
            if j in used_gt:
                continue
            iou = compute_miou(pred_mask, gt_mask)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_j = j
        if best_j != -1:
            matches[i] = best_j
            used_gt.add(best_j)
    return matches


def compute_overall_contour_accuracy(
    pred_multiclass, gt_multiclass, metric="hausdorff"
):
    """Вычислить среднее контурное расстояние для всех сопоставленных объектов."""
    pred_binary_list = get_binary_masks_from_multiclass(pred_multiclass)
    gt_binary_list = get_binary_masks_from_multiclass(gt_multiclass)

    if not pred_binary_list or not gt_binary_list:
        return float("inf")  # или 0, если нет объектов

    matches = match_objects_by_iou(pred_binary_list, gt_binary_list)

    distances = []
    for pred_idx, gt_idx in matches.items():
        dist = compute_contour_accuracy_for_pair(
            pred_binary_list[pred_idx], gt_binary_list[gt_idx], metric
        )
        distances.append(dist)

    return np.mean(distances) if distances else 0.0
