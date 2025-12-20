import cv2
import numpy as np
import math


def _to_xyxy(box):
    """
    Конвертирует [x, y, w, h] в [x1, y1, x2, y2].
    """
    if box is None or len(box) < 4:
        return None
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def compute_bbox_iou(pred_box, gt_box):
    """
    Считает Intersection over Union (IoU) для двух рамок формата [x, y, w, h].
    """
    if pred_box is None or gt_box is None:
        return 0.0

    # Конвертируем в x1, y1, x2, y2 для удобства расчета
    b1 = _to_xyxy(pred_box)
    b2 = _to_xyxy(gt_box)

    # Координаты пересечения
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    # Площадь пересечения
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Площади самих рамок
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    # Площадь объединения
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_center_distance(pred_box, gt_box):
    """
    Считает Евклидово расстояние между центрами рамок (в пикселях).
    Меньше = лучше.
    """
    if pred_box is None or gt_box is None:
        return -1.0  # Ошибка

    # Центр = x + w/2, y + h/2
    cx_pred = pred_box[0] + pred_box[2] / 2
    cy_pred = pred_box[1] + pred_box[3] / 2

    cx_gt = gt_box[0] + gt_box[2] / 2
    cy_gt = gt_box[1] + gt_box[3] / 2

    dist = math.sqrt((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2)
    return dist


def compute_aspect_ratio_error(pred_box, gt_box):
    """
    Считает разницу в соотношении сторон (Aspect Ratio).
    Возвращает абсолютную разницу. 0 = идеально.
    """
    if pred_box is None or gt_box is None:
        return -1.0

    # w / h
    ar_pred = pred_box[2] / (pred_box[3] + 1e-6)
    ar_gt = gt_box[2] / (gt_box[3] + 1e-6)

    return abs(ar_pred - ar_gt)


def get_box_for_id(mask_indices, obj_id):
    """
    Извлекает bounding box [x, y, w, h] для конкретного id объекта из маски индексов.
    """
    # Создаем бинарную маску только для этого объекта
    binary_mask = (mask_indices == obj_id).astype(np.uint8)

    # Находим контуры
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Берем самый большой контур (на случай шума)
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)  # Возвращает x, y, w, h


class BoxEvaluator:
    """
    Класс для накопления статистики и расчета F1-score, Precision, Recall и IoU для рамок.
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.ious = []
        self.dists = []

    def update(self, pred_box, gt_box):
        """
        Обновляет метрики для одной пары (предсказание, эталон).
        Оба аргумента могут быть None (если объект не найден или лишний).
        """
        if gt_box is not None and pred_box is not None:
            # 1. Оба существуют: проверяем качество совпадения
            iou = compute_bbox_iou(pred_box, gt_box)
            dist = compute_center_distance(pred_box, gt_box)

            self.ious.append(iou)
            self.dists.append(dist)

            if iou >= self.iou_threshold:
                self.tp += 1
            else:
                # Объект найден, но IoU слишком низкий -> Ложное срабатывание + Пропуск
                self.fp += 1
                self.fn += 1

        elif gt_box is not None and pred_box is None:
            # 2. Объект есть в GT, но трекер его не нашел (False Negative)
            self.fn += 1
            self.ious.append(0.0)  # Для честного среднего IoU считаем это нулем

        elif gt_box is None and pred_box is not None:
            # 3. Трекер нашел объект, которого нет в GT (False Positive)
            self.fp += 1

    def compute(self):
        """
        Возвращает словарь с итоговыми метриками.
        """
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        mean_iou = np.mean(self.ious) if self.ious else 0.0
        mean_dist = np.mean(self.dists) if self.dists else 0.0

        return {
            "mean_iou": mean_iou,
            "mean_dist": mean_dist,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }
