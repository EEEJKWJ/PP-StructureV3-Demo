"""表格裁切、局部增强和坐标变换。"""

import math

import cv2
import numpy as np


def crop_table_from_image(image, table_bbox, padding=8):
    """从页面图像裁切表格并记录可逆坐标信息。"""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("页面图像为空")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in table_bbox[:4]]
    crop_x1 = max(0, int(math.floor(x1)) - padding)
    crop_y1 = max(0, int(math.floor(y1)) - padding)
    crop_x2 = min(width, int(math.ceil(x2)) + padding)
    crop_y2 = min(height, int(math.ceil(y2)) + padding)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise ValueError("表格裁切区域为空")
    return crop, {
        "page_width": width,
        "page_height": height,
        "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
        "padding": padding,
    }


def estimate_skew_angle(image):
    """根据较长横线估计小幅表格倾斜角度。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    height, width = binary.shape[:2]
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=max(30, width // 10),
        minLineLength=max(40, width // 5),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [float(value) for value in line]
        if x2 == x1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= 7:
            angles.append(angle)
    if not angles:
        return 0.0
    angle = float(np.median(angles))
    return angle if abs(angle) >= 0.25 else 0.0


def _scaled_matrix(scale):
    return np.array(
        [[scale, 0, 0], [0, scale, 0]],
        dtype=np.float64,
    )


def _rotation_matrix(shape, angle):
    height, width = shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return matrix, new_width, new_height


def enhance_table_crop(
    crop,
    target_long_side=1200,
    max_scale=3.0,
    deskew=True,
):
    """执行表格局部增强并返回局部到裁切图的仿射矩阵。"""
    if crop is None or crop.size == 0:
        raise ValueError("表格裁切图像为空")
    height, width = crop.shape[:2]
    long_side = max(width, height)
    scale = 1.0
    if long_side < target_long_side:
        scale = min(target_long_side / long_side, max_scale)

    matrix = _scaled_matrix(scale)
    enhanced = (
        cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        if scale != 1.0
        else crop.copy()
    )

    angle = estimate_skew_angle(enhanced) if deskew else 0.0
    if angle != 0.0:
        rotation, new_width, new_height = _rotation_matrix(
            enhanced.shape,
            -angle,
        )
        enhanced = cv2.warpAffine(
            enhanced,
            rotation,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        matrix = np.vstack([rotation, [0, 0, 1]]) @ np.vstack(
            [matrix, [0, 0, 1]]
        )
        matrix = matrix[:2]

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    enhanced = cv2.addWeighted(
        enhanced,
        1.12,
        blurred,
        -0.12,
        0,
    )
    return enhanced, {
        "matrix": matrix.tolist(),
        "inverse_matrix": cv2.invertAffineTransform(matrix).tolist(),
        "scale": scale,
        "skew_angle": angle,
        "source_width": width,
        "source_height": height,
        "output_width": enhanced.shape[1],
        "output_height": enhanced.shape[0],
    }


def map_bbox(bbox, matrix):
    """使用 2x3 仿射矩阵转换矩形框。"""
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    points = np.array(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        dtype=np.float64,
    )
    transformed = cv2.transform(
        points,
        np.asarray(matrix, dtype=np.float64),
    )[0]
    return [
        round(float(transformed[:, 0].min())),
        round(float(transformed[:, 1].min())),
        round(float(transformed[:, 0].max())),
        round(float(transformed[:, 1].max())),
    ]


def processed_bbox_to_page(bbox, crop_info, transform):
    """把增强图坐标映射回页面坐标。"""
    crop_bbox = map_bbox(bbox, transform["inverse_matrix"])
    x1, y1, _, _ = crop_info["crop_bbox"]
    return [
        max(0, crop_bbox[0] + x1),
        max(0, crop_bbox[1] + y1),
        crop_bbox[2] + x1,
        crop_bbox[3] + y1,
    ]


def page_bbox_to_processed(bbox, crop_info, transform):
    """把页面坐标映射到增强表格图坐标。"""
    crop_x1, crop_y1, _, _ = crop_info["crop_bbox"]
    local = [
        bbox[0] - crop_x1,
        bbox[1] - crop_y1,
        bbox[2] - crop_x1,
        bbox[3] - crop_y1,
    ]
    return map_bbox(local, transform["matrix"])
