import os
import cv2
import numpy as np
import json

# Ð´Ð»Ñ ÑÐ¸Ð½Ð¸Ñ… ÐºÐ¾ÑÑ‚ÐµÐ¹ Ð¿Ð¾Ð´Ð¾Ð±Ñ€Ð°Ð» ÑÑ‚Ð¸ hsv
# hsv_min = np.array((45, 49, 50), np.uint8)
# hsv_max = np.array((255, 255, 255), np.uint8)

hsv_min = np.array((0, 40, 0), np.uint8)
hsv_max = np.array((255, 255, 255), np.uint8)


def get_contour_by_hsv(image, hsv_minimum, hsv_maximum):
    image = cv2.GaussianBlur(image, (11, 11), 0)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thr = cv2.inRange(hsv_img, hsv_minimum, hsv_maximum, cv2.THRESH_BINARY)
    cnt, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return cnt


def draw_contours(image, cnt, on_image=True):
    if on_image:
        cv2.drawContours(image, cnt, -1, (0, 0, 255), 2)
        return image
    image_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))
    cv2.drawContours(image_contours, cnt, -1, (0, 0, 255), 1)
    return image_contours


def contour_x_coordinate(contour):
    return [int(x) for (x, y) in contour.reshape(-1, 2)]


def contour_y_coordinate(contour):
    return [int(y) for (x, y) in contour.reshape(-1, 2)]


def append_to_json(_data, path):
    with open(path, 'ab+') as file:
        file.seek(0, 2)
        if file.tell() == 0:
            file.write(json.dumps(_data).encode())
        else:
            file.seek(-1, 2)
            file.truncate()
            file.write('  ,'.encode())
            file.write(json.dumps(_data).encode()[1:])


def contour_by_hsv(image, hsv_minimum, hsv_maximum, cnt_hierarchy):
    image = cv2.GaussianBlur(image, (9, 9), 0)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thr = cv2.inRange(hsv_img, hsv_minimum, hsv_maximum, cv2.THRESH_BINARY)
    cnt = cv2.findContours(thr, cnt_hierarchy, cv2.CHAIN_APPROX_NONE)[0]
    return cnt


def draw_contours(image, contours, color, on_image=True):
    empty_shape = np.uint8(np.zeros((image.shape[0], image.shape[1])))
    # empty_shape += 255
    if on_image:
        cv2.drawContours(image, contours, -1, color, 1)
        return image
    cv2.drawContours(empty_shape, contours, -1, color, 1)
    return empty_shape


def max_contour_to_json(folder_path):
    hsv_min = np.array((45, 49, 50), np.uint8)
    hsv_max = np.array((255, 255, 255), np.uint8)
    for i, filename in enumerate(os.listdir(folder_path)):
        img = cv2.imread(f'{folder_path}/{filename}')
        contour = max(
            contour_by_hsv(img, hsv_min, hsv_max, cv2.RETR_EXTERNAL),
            key=cv2.contourArea,
        )
        data = {
            f'{filename}{img.size}': {
                'fileref': '',
                'size': img.size,
                'filename': filename,
                'base64_img_data': '',
                'file_attributes': {},
                'regions': {
                    '0': {
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': contour_x_coordinate(contour),
                            'all_points_y': contour_y_coordinate(contour),
                        },
                        'region_attributes': {},
                    }
                },
            }
        }
        append_to_json(data, 'bones.json')


# hsv_min(26,35,0)hsv_max(62,255,255)yellow
def contours_to_json(folder_path):
    hsv_min = np.array((26, 35, 0), np.uint8)
    hsv_max = np.array((62, 255, 255), np.uint8)
    for i, filename in enumerate(os.listdir(folder_path)):
        img = cv2.imread(f'{folder_path}/{filename}')
        contours = contour_by_hsv(img, hsv_min, hsv_max, cv2.RETR_EXTERNAL)
        regions = {}
        for i in range(len(contours)):
            regions.update(
                {
                    f'{i}': {
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': contour_x_coordinate(contours[i]),
                            'all_points_y': contour_y_coordinate(contours[i]),
                        },
                        'region_attributes': {},
                    }
                }
            )
        data = {
            f'{filename}{img.size}': {
                'fileref': '',
                'size': img.size,
                'filename': filename,
                'base64_img_data': '',
                'file_attributes': {},
                'regions': regions,
            }
        }
        append_to_json(data, 'bones.json')
