"""估计原图与官方预处理页面之间的几何对应关系。"""

import copy

import cv2
import numpy as np


_IDENTITY_TOLERANCE = 1e-6


def _read_gray(image):
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("页面图像为空")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _identity_matrix():
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform_bbox(bbox, matrix, width, height):
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    corners = np.array(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        dtype=np.float64,
    )
    transformed = cv2.perspectiveTransform(corners, matrix)[0]
    return [
        round(float(np.clip(transformed[:, 0].min(), 0, width - 1))),
        round(float(np.clip(transformed[:, 1].min(), 0, height - 1))),
        round(float(np.clip(transformed[:, 0].max(), 0, width))),
        round(float(np.clip(transformed[:, 1].max(), 0, height))),
    ]


def _alignment_result(
    matrix,
    method,
    source_shape,
    target_shape,
    match_count=0,
    inlier_count=0,
    mean_error=None,
    status="available",
):
    inverse = np.linalg.inv(matrix)
    return {
        "status": status,
        "method": method,
        "matrix": matrix.tolist(),
        "inverse_matrix": inverse.tolist(),
        "source_width": int(source_shape[1]),
        "source_height": int(source_shape[0]),
        "target_width": int(target_shape[1]),
        "target_height": int(target_shape[0]),
        "match_count": int(match_count),
        "inlier_count": int(inlier_count),
        "inlier_ratio": (
            round(inlier_count / match_count, 4)
            if match_count
            else None
        ),
        "mean_reprojection_error": (
            round(float(mean_error), 4)
            if mean_error is not None
            else None
        ),
    }


def estimate_page_alignment(source_image, target_image):
    """估计原图到预处理页面的单应变换。

    官方文档矫正只返回处理后的图像，不返回坐标矩阵。这里使用页面特征
    匹配恢复几何变换。若页面像素完全一致，则直接返回单位矩阵。若特征
    匹配证据不足，则明确标记失败，避免用错误坐标计算指标。
    """
    source_gray = _read_gray(source_image)
    target_gray = _read_gray(target_image)
    if source_gray.shape == target_gray.shape:
        difference = cv2.absdiff(source_gray, target_gray)
        if float(difference.mean()) <= _IDENTITY_TOLERANCE:
            return _alignment_result(
                _identity_matrix(),
                "identity",
                source_gray.shape,
                target_gray.shape,
            )

    cv2.setRNGSeed(0)
    best = None
    detectors = (
        ("sift-ransac", cv2.SIFT_create()),
        (
            "orb-ransac",
            cv2.ORB_create(
                nfeatures=4000,
                scaleFactor=1.2,
                nlevels=8,
                fastThreshold=10,
            ),
        ),
    )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for method, detector in detectors:
        source_points, source_descriptors = detector.detectAndCompute(
            source_gray,
            None,
        )
        target_points, target_descriptors = detector.detectAndCompute(
            target_gray,
            None,
        )
        if (
            source_descriptors is None
            or target_descriptors is None
            or len(source_points) < 4
            or len(target_points) < 4
        ):
            continue
        if detector.descriptorType() == cv2.CV_8U:
            use_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            use_matcher = matcher
        pairs = use_matcher.knnMatch(
            source_descriptors,
            target_descriptors,
            k=2,
        )
        good = [
            first
            for first, second in pairs
            if first.distance < 0.75 * max(second.distance, 1e-6)
        ]
        if len(good) < 8:
            continue
        source_xy = np.float32([
            source_points[item.queryIdx].pt for item in good
        ])
        target_xy = np.float32([
            target_points[item.trainIdx].pt for item in good
        ])
        matrix, mask = cv2.findHomography(
            source_xy,
            target_xy,
            cv2.RANSAC,
            4.0,
            maxIters=3000,
            confidence=0.999,
        )
        if matrix is None or mask is None:
            continue
        inliers = mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < 12:
            continue
        projected = cv2.perspectiveTransform(
            source_xy[inliers].reshape(-1, 1, 2),
            matrix,
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - target_xy[inliers], axis=1)
        mean_error = float(errors.mean())
        inlier_ratio = inlier_count / len(good)
        if inlier_ratio < 0.12 or mean_error > 8.0:
            continue
        candidate = {
            "matrix": matrix,
            "method": method,
            "match_count": len(good),
            "inlier_count": inlier_count,
            "mean_error": mean_error,
            "score": inlier_count - mean_error * 2,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        return _alignment_result(
            _identity_matrix(),
            "none",
            source_gray.shape,
            target_gray.shape,
            status="failed",
        )
    return _alignment_result(
        best["matrix"],
        best["method"],
        source_gray.shape,
        target_gray.shape,
        match_count=best["match_count"],
        inlier_count=best["inlier_count"],
        mean_error=best["mean_error"],
    )


def page_bbox_to_aligned(bbox, alignment):
    """把原图 bbox 转换到预处理页面坐标。"""
    return _transform_bbox(
        bbox,
        np.asarray(alignment["matrix"], dtype=np.float64),
        int(alignment["target_width"]),
        int(alignment["target_height"]),
    )


def aligned_bbox_to_page(bbox, alignment):
    """把预处理页面 bbox 转换回原图坐标。"""
    return _transform_bbox(
        bbox,
        np.asarray(alignment["inverse_matrix"], dtype=np.float64),
        int(alignment["source_width"]),
        int(alignment["source_height"]),
    )


def align_ground_truth(ground_truth, alignment):
    """复制真值并把单元格和表格框转换到预处理页面坐标。"""
    aligned = copy.deepcopy(ground_truth)
    if alignment.get("status") != "available":
        return aligned
    table_bbox = aligned.get("table_bbox")
    if table_bbox:
        aligned["table_bbox"] = page_bbox_to_aligned(
            table_bbox,
            alignment,
        )
    for cell in aligned.get("cells") or []:
        if cell.get("bbox"):
            cell["bbox"] = page_bbox_to_aligned(
                cell["bbox"],
                alignment,
            )
    return aligned
