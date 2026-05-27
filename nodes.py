from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image


OUTER_MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
]
INNER_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87]
MOUTH_SHAPES = ("closed", "half", "open", "e", "u")
DEFAULT_ANGLE_RANGE_DEGREES = 45
DEFAULT_ANGLE_STEP_DEGREES = 15
MIN_EXTRACTED_OPEN_RANGE = 0.12
VIDEO_EXTENSIONS = (".webm", ".mp4", ".mkv", ".gif", ".mov", ".m4v")


@dataclass
class MouthRecord:
    valid: bool
    mouth_bbox: tuple[float, float, float, float] | None = None
    track_quad: list[list[float]] | None = None
    open_ratio: float = 0.0
    width: float = 0.0
    height: float = 0.0
    quality_score: float = 0.0
    occlusion_score: float = 1.0
    signal_ratio: float = 0.0
    mouth_likeness_score: float = 0.0
    source: str = "missing"


def _comfy_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _node_root() -> Path:
    return Path(__file__).resolve().parent


def _comfy_input_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_input_directory())
    except Exception:
        return Path.cwd()


def _comfy_output_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd()


def _resolve_path(value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw, _comfy_input_dir() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _input_video_files() -> list[str]:
    input_dir = _comfy_input_dir()
    if not input_dir.exists():
        return [""]
    videos = [
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return [""] + sorted(videos)


def _safe_asset_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned or "pngtuber_video_mouth"


def _quad_from_bbox(
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
    width: int,
    height: int,
    angle_rad: float = 0.0,
) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bw = max(8.0, x1 - x0)
    bh = max(8.0, y1 - y0)
    side = max(bw, bh) * scale
    half = side * 0.5
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return [
        [
            min(float(width - 1), max(0.0, cx + dx * cos_a - dy * sin_a)),
            min(float(height - 1), max(0.0, cy + dx * sin_a + dy * cos_a)),
        ]
        for dx, dy in corners
    ]


def _mouth_angle_from_points(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    arr = points.astype(np.float32)
    arr = arr - arr.mean(axis=0, keepdims=True)
    if float(np.abs(arr).sum()) <= 1e-3:
        return 0.0
    _, _, vh = np.linalg.svd(arr, full_matrices=False)
    vx, vy = vh[0]
    angle = math.atan2(float(vy), float(vx))
    while angle > math.pi / 2:
        angle -= math.pi
    while angle < -math.pi / 2:
        angle += math.pi
    return max(math.radians(-35.0), min(math.radians(35.0), angle))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mouth_signal_mask(crop_bgr: np.ndarray, *, rx_scale: float = 0.40, ry_scale: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    yy, xx = np.ogrid[:crop_bgr.shape[0], :crop_bgr.shape[1]]
    cy, cx = crop_bgr.shape[0] * 0.5, crop_bgr.shape[1] * 0.5
    rx, ry = max(1.0, crop_bgr.shape[1] * rx_scale), max(1.0, crop_bgr.shape[0] * ry_scale)
    mouth_window = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    edges = cv2.Canny(gray, 80, 170) > 0
    dark = val < 100
    red_lip = ((hue < 12) | (hue > 164)) & (sat > 58) & (val < 235) & (val > 30)
    mask = (dark | red_lip | (edges & (val < 205))) & mouth_window
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8)).astype(bool)
    return mask, mouth_window


def _best_mouth_component(mask: np.ndarray, crop_bgr: np.ndarray, mouth_window: np.ndarray) -> tuple[np.ndarray, dict[str, float]] | None:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return None

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)
    red_lip = ((hue < 14) | (hue > 162)) & (sat > 45) & (val < 245) & (val > 25)
    dark = val < 95
    bad_chroma = (sat > 55) & (val > 45) & ~red_lip
    h, w = mask.shape
    center = np.array([w * 0.5, h * 0.50])
    window_area = max(1, int(mouth_window.sum()))

    best_label = 0
    best_score = -1e18
    best_meta: dict[str, float] = {}
    for label in range(1, num):
        x, y, w_, h_, area = stats[label].tolist()
        if area < max(4, window_area * 0.0012):
            continue
        if area > window_area * 0.34:
            continue
        aspect = w_ / max(1, h_)
        if aspect < 0.55 or aspect > 16.0:
            continue
        cx, cy = centroids[label]
        nx = abs((float(cx) / max(1, w)) - 0.5)
        ny = abs((float(cy) / max(1, h)) - 0.5)
        if nx > 0.44 or ny > 0.42:
            continue

        component = labels == label
        lip_ratio = float((component & red_lip).sum()) / max(1, area)
        dark_ratio = float((component & dark).sum()) / max(1, area)
        bad_chroma_ratio = float((component & bad_chroma).sum()) / max(1, area)
        if bad_chroma_ratio > 0.34 and lip_ratio < 0.08 and dark_ratio < 0.70:
            continue
        dist = float(np.linalg.norm(np.array([cx, cy]) - center)) / max(1.0, min(w, h))
        thin_bonus = min(aspect, 6.0) * 0.08
        chroma_bonus = lip_ratio * 2.8 + dark_ratio * 0.65
        size_score = min(float(area) / max(1.0, window_area * 0.055), 1.0)
        score = chroma_bonus + size_score + thin_bonus - bad_chroma_ratio * 1.7 - dist * 1.65 - max(0.0, h_ / max(1, h) - 0.42) * 2.2
        if score > best_score:
            best_score = score
            best_label = label
            best_meta = {
                "area": float(area),
                "aspect": float(aspect),
                "lip_ratio": lip_ratio,
                "dark_ratio": dark_ratio,
                "bad_chroma_ratio": bad_chroma_ratio,
                "score": float(score),
                "x": float(x),
                "y": float(y),
                "w": float(w_),
                "h": float(h_),
            }
    if best_label == 0 or best_score < 0.38:
        return None
    return labels == best_label, best_meta


def _score_mouth_candidate(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    open_ratio: float,
    mouth_width: float,
    mouth_height: float,
) -> tuple[float, float, float, float]:
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _scaled_bbox(bbox, scale=1.45, width=width, height=height)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 1.0, 0.0, 0.0

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    if crop_bgr.size == 0:
        return 0.0, 1.0, 0.0, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask, mouth_window = _mouth_signal_mask(crop_bgr)
    window_area = max(1, int(mouth_window.sum()))
    component_result = _best_mouth_component(mask, crop_bgr, mouth_window)
    component_meta: dict[str, float] = {}
    if component_result is not None:
        component_mask, component_meta = component_result
        signal_ratio = float(component_mask.sum()) / window_area
    else:
        signal_ratio = float(mask.sum()) / window_area
    edges = (cv2.Canny(gray, 80, 170) > 0) & mouth_window
    edge_ratio = float(edges.sum()) / window_area
    contrast = float(gray[mouth_window].std()) / 64.0 if window_area > 0 else 0.0

    aspect = mouth_width / max(1.0, mouth_height)
    aspect_penalty = 0.0
    if aspect < 1.1:
        aspect_penalty = (1.1 - aspect) * 0.35
    elif aspect > 12.0:
        aspect_penalty = (aspect - 12.0) * 0.04

    too_small = max(0.0, (0.012 - signal_ratio) / 0.012)
    too_large = max(0.0, (signal_ratio - 0.42) / 0.42)
    extreme_open_penalty = max(0.0, open_ratio - 0.55) * 0.25
    mouth_likeness = max(0.0, component_meta.get("score", 0.0)) / 4.0
    quality = (
        0.08
        + signal_ratio * 2.15
        + edge_ratio * 1.55
        + min(contrast, 1.4) * 0.24
        + mouth_likeness * 0.55
        - too_small * 0.55
        - too_large * 0.45
        - aspect_penalty
        - extreme_open_penalty
    )
    if component_result is None:
        quality *= 0.45
    occlusion = _clamp_float(too_large * 0.65 + too_small * 0.45 + aspect_penalty * 0.35 + (0.22 if component_result is None else 0.0), 0.0, 1.0)
    return _clamp_float(quality, 0.0, 1.0), occlusion, signal_ratio, _clamp_float(mouth_likeness, 0.0, 1.0)


def _with_quality(record: MouthRecord, frame_bgr: np.ndarray) -> MouthRecord:
    if not record.valid or record.mouth_bbox is None:
        return record
    quality, occlusion, signal_ratio, mouth_likeness = _score_mouth_candidate(
        frame_bgr,
        record.mouth_bbox,
        open_ratio=record.open_ratio,
        mouth_width=record.width,
        mouth_height=record.height,
    )
    record.quality_score = quality
    record.occlusion_score = occlusion
    record.signal_ratio = signal_ratio
    record.mouth_likeness_score = mouth_likeness
    return record


def _refine_mouth_signal_bbox(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], float, float]:
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _scaled_bbox(bbox, scale=1.35, width=width, height=height)
    if x1 <= x0 or y1 <= y0:
        return bbox, 0.08, 0.0

    crop = frame_bgr[y0:y1, x0:x1]
    mask, mouth_window = _mouth_signal_mask(crop, rx_scale=0.44, ry_scale=0.30)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 4), np.uint8))
    component_result = _best_mouth_component(mask, crop, mouth_window)
    if component_result is None:
        return bbox, 0.08, _estimate_mouth_angle_from_frame(frame_bgr, bbox)

    component, meta = component_result
    ys, xs = np.nonzero(component)
    if len(xs) >= 4:
        points = np.column_stack([xs + x0, ys + y0])
        angle_rad = _mouth_angle_from_points(points)
    else:
        angle_rad = 0.0

    component_open = meta["h"] / max(1.0, meta["w"])
    open_ratio = _clamp_float(component_open, 0.015, 0.65)
    return bbox, open_ratio, angle_rad


def _estimate_mouth_angle_from_frame(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> float:
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _scaled_bbox(bbox, scale=1.55, width=width, height=height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = frame_bgr[y0:y1, x0:x1]
    mask, _ = _mouth_signal_mask(crop, rx_scale=0.42, ry_scale=0.28)
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return 0.0
    points = np.column_stack([xs + x0, ys + y0])
    return _mouth_angle_from_points(points)


def _bbox_from_quad(quad: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def _detect_mouth(
    frame_bgr: np.ndarray,
    face_mesh: Any,
    *,
    min_confidence: float,
    quad_scale: float,
) -> MouthRecord:
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    faces = getattr(result, "multi_face_landmarks", None)
    if not faces:
        return MouthRecord(valid=False)

    best: MouthRecord | None = None
    best_area = 0.0
    for face in faces:
        pts = []
        for idx in OUTER_MOUTH + INNER_MOUTH:
            lm = face.landmark[idx]
            pts.append((float(lm.x) * width, float(lm.y) * height))
        arr = np.asarray(pts, dtype=np.float32)
        x0, y0 = np.percentile(arr[:, 0], 2), np.percentile(arr[:, 1], 2)
        x1, y1 = np.percentile(arr[:, 0], 98), np.percentile(arr[:, 1], 98)
        x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
        x1, y1 = min(float(width - 1), float(x1)), min(float(height - 1), float(y1))
        mouth_w = max(1.0, x1 - x0)
        mouth_h = max(1.0, y1 - y0)
        upper = face.landmark[13]
        lower = face.landmark[14]
        left = face.landmark[61]
        right = face.landmark[291]
        vertical = abs(float(lower.y - upper.y)) * height
        horizontal = max(1.0, abs(float(right.x - left.x)) * width)
        open_ratio = float(vertical / horizontal)
        mouth_angle = math.atan2(
            (float(right.y - left.y) * height),
            (float(right.x - left.x) * width),
        )
        area = mouth_w * mouth_h
        record = MouthRecord(
            valid=True,
            mouth_bbox=(x0, y0, x1, y1),
            track_quad=_quad_from_bbox(
                (x0, y0, x1, y1),
                scale=quad_scale,
                width=width,
                height=height,
                angle_rad=mouth_angle,
            ),
            open_ratio=open_ratio,
            width=mouth_w,
            height=mouth_h,
            source=f"mediapipe:{min_confidence:.2f}",
        )
        if area > best_area:
            best = record
            best_area = area
    return best or MouthRecord(valid=False)


def _face_yolo_mouth_fallback(
    frame_bgr: np.ndarray,
    model: Any | None,
    *,
    quad_scale: float,
) -> MouthRecord:
    if model is None:
        return MouthRecord(valid=False)
    height, width = frame_bgr.shape[:2]
    try:
        results = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.25)
    except Exception:
        return MouthRecord(valid=False)
    best_box: tuple[float, float, float, float] | None = None
    best_area = 0.0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            continue
        for row in xyxy.cpu().numpy().tolist():
            x0, y0, x1, y1 = map(float, row[:4])
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if area > best_area:
                best_area = area
                best_box = (x0, y0, x1, y1)
    if best_box is None:
        return MouthRecord(valid=False)
    fx0, fy0, fx1, fy1 = best_box
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    cx = fx0 + fw * 0.5
    cy = fy0 + fh * 0.68
    mouth_w = fw * 0.34
    mouth_h = fh * 0.12
    bbox = (
        max(0.0, cx - mouth_w * 0.5),
        max(0.0, cy - mouth_h * 0.5),
        min(float(width - 1), cx + mouth_w * 0.5),
        min(float(height - 1), cy + mouth_h * 0.5),
    )
    return MouthRecord(
        valid=True,
        mouth_bbox=bbox,
        track_quad=_quad_from_bbox(bbox, scale=quad_scale, width=width, height=height),
        open_ratio=0.08,
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        source="face_yolo_fallback",
    )


def _load_anime_cascade() -> Any | None:
    cascade_path = _node_root() / "models" / "lbpcascade_animeface.xml"
    if not cascade_path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return None
    return cascade


def _refine_anime_mouth_bbox(
    frame_bgr: np.ndarray,
    face_box: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], float, float]:
    height, width = frame_bgr.shape[:2]
    fx0, fy0, fx1, fy1 = face_box
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    rx0 = int(max(0, round(fx0 + fw * 0.18)))
    rx1 = int(min(width - 1, round(fx1 - fw * 0.18)))
    ry0 = int(max(0, round(fy0 + fh * 0.54)))
    ry1 = int(min(height - 1, round(fy0 + fh * 0.84)))
    def estimated() -> tuple[tuple[float, float, float, float], float, float]:
        cx = fx0 + fw * 0.5
        cy = fy0 + fh * 0.69
        bbox = (cx - fw * 0.16, cy - fh * 0.04, cx + fw * 0.16, cy + fh * 0.04)
        return _refine_mouth_signal_bbox(frame_bgr, bbox)

    if rx1 <= rx0 or ry1 <= ry0:
        return estimated()

    roi = frame_bgr[ry0:ry1, rx0:rx1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    edges = cv2.Canny(gray, 50, 150)
    dark = val < 125
    color_line = (sat > 55) & (val < 210)
    mask = (dark | color_line | (edges > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best: tuple[int, int, int, int, int] | None = None
    best_score = -1e18
    roi_h, roi_w = mask.shape
    center = np.array([roi_w * 0.5, roi_h * 0.46])
    for label in range(1, num):
        x, y, w_, h_, area = stats[label].tolist()
        if area < max(3, roi_w * roi_h * 0.0008):
            continue
        aspect = w_ / max(1, h_)
        if aspect < 0.7 or aspect > 18.0:
            continue
        cx, cy = centroids[label]
        dist = float(np.linalg.norm(np.array([cx, cy]) - center))
        score = float(area) + min(aspect, 6.0) * 6.0 - dist * 1.5
        if score > best_score:
            best_score = score
            best = (label, x, y, w_, h_)
    if best is None:
        return estimated()
    label, x, y, w_, h_ = best
    component = labels[y : y + h_, x : x + w_] == label
    ys, xs = np.nonzero(component)
    if len(xs) >= 4:
        angle_points = np.column_stack([xs + x + rx0, ys + y + ry0])
        angle_rad = _mouth_angle_from_points(angle_points)
    else:
        angle_rad = 0.0
    pad_x = max(2.0, w_ * 0.35)
    pad_y = max(2.0, h_ * 0.85)
    x0 = max(0.0, rx0 + x - pad_x)
    y0 = max(0.0, ry0 + y - pad_y)
    x1 = min(float(width - 1), rx0 + x + w_ + pad_x)
    y1 = min(float(height - 1), ry0 + y + h_ + pad_y)
    if (x1 - x0) > fw * 0.46 or (y1 - y0) > fh * 0.24:
        return estimated()
    open_ratio = _clamp_float(float(h_ / max(1.0, w_)), 0.015, 0.65)
    return (x0, y0, x1, y1), open_ratio, angle_rad


def _anime_cascade_mouth_fallback(
    frame_bgr: np.ndarray,
    cascade: Any | None,
    *,
    quad_scale: float,
) -> MouthRecord:
    if cascade is None:
        return MouthRecord(valid=False)
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(32, 32),
    )
    if len(faces) == 0:
        return MouthRecord(valid=False)
    x, y, w_, h_ = max(faces, key=lambda item: item[2] * item[3])
    face_box = (float(x), float(y), float(x + w_), float(y + h_))
    mouth_bbox, open_ratio, mouth_angle = _refine_anime_mouth_bbox(frame_bgr, face_box)
    return MouthRecord(
        valid=True,
        mouth_bbox=mouth_bbox,
        track_quad=_quad_from_bbox(
            mouth_bbox,
            scale=quad_scale,
            width=width,
            height=height,
            angle_rad=mouth_angle,
        ),
        open_ratio=open_ratio,
        width=mouth_bbox[2] - mouth_bbox[0],
        height=mouth_bbox[3] - mouth_bbox[1],
        source="anime_cascade_mouth_refine",
    )


def _load_face_yolo_model() -> Any | None:
    model_path = _comfy_root_from_this_file() / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt"
    if not model_path.exists():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(model_path))
    except Exception:
        return None


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interpolate_records(records: list[MouthRecord]) -> list[MouthRecord]:
    valid_indices = [idx for idx, rec in enumerate(records) if rec.valid and rec.mouth_bbox and rec.track_quad]
    if not valid_indices:
        raise RuntimeError("No mouth landmarks were detected. Use a clearer frontal face video.")

    first = valid_indices[0]
    last_valid = first
    for idx in range(0, first):
        records[idx] = records[first]

    for next_valid in valid_indices[1:]:
        prev = records[last_valid]
        nxt = records[next_valid]
        span = next_valid - last_valid
        for off in range(1, span):
            t = off / span
            bbox = tuple(_lerp(prev.mouth_bbox[i], nxt.mouth_bbox[i], t) for i in range(4))  # type: ignore[index]
            quad = [
                [_lerp(prev.track_quad[p][0], nxt.track_quad[p][0], t), _lerp(prev.track_quad[p][1], nxt.track_quad[p][1], t)]  # type: ignore[index]
                for p in range(4)
            ]
            records[last_valid + off] = MouthRecord(
                valid=True,
                mouth_bbox=bbox,  # type: ignore[arg-type]
                track_quad=quad,
                open_ratio=_lerp(prev.open_ratio, nxt.open_ratio, t),
                width=_lerp(prev.width, nxt.width, t),
                height=_lerp(prev.height, nxt.height, t),
                quality_score=_lerp(prev.quality_score, nxt.quality_score, t),
                occlusion_score=_lerp(prev.occlusion_score, nxt.occlusion_score, t),
                signal_ratio=_lerp(prev.signal_ratio, nxt.signal_ratio, t),
                mouth_likeness_score=_lerp(prev.mouth_likeness_score, nxt.mouth_likeness_score, t),
                source="interpolated",
            )
        last_valid = next_valid

    for idx in range(valid_indices[-1] + 1, len(records)):
        records[idx] = records[valid_indices[-1]]

    smoothed: list[MouthRecord] = []
    alpha = 0.35
    prev = records[0]
    smoothed.append(prev)
    for rec in records[1:]:
        bbox = tuple(_lerp(prev.mouth_bbox[i], rec.mouth_bbox[i], alpha) for i in range(4))  # type: ignore[index]
        quad = [
            [_lerp(prev.track_quad[p][0], rec.track_quad[p][0], alpha), _lerp(prev.track_quad[p][1], rec.track_quad[p][1], alpha)]  # type: ignore[index]
            for p in range(4)
        ]
        prev = MouthRecord(
            valid=True,
            mouth_bbox=bbox,  # type: ignore[arg-type]
            track_quad=quad,
            open_ratio=_lerp(prev.open_ratio, rec.open_ratio, alpha),
            width=_lerp(prev.width, rec.width, alpha),
            height=_lerp(prev.height, rec.height, alpha),
            quality_score=_lerp(prev.quality_score, rec.quality_score, alpha),
            occlusion_score=_lerp(prev.occlusion_score, rec.occlusion_score, alpha),
            signal_ratio=_lerp(prev.signal_ratio, rec.signal_ratio, alpha),
            mouth_likeness_score=_lerp(prev.mouth_likeness_score, rec.mouth_likeness_score, alpha),
            source=rec.source,
        )
        smoothed.append(prev)
    return smoothed


def _eligible_record_indices(records: list[MouthRecord], *, occlusion_filter: bool) -> list[int]:
    valid = [idx for idx, rec in enumerate(records) if rec.valid and rec.mouth_bbox and rec.track_quad]
    preferred_sources = [
        idx
        for idx in valid
        if records[idx].source != "face_yolo_fallback" and records[idx].source != "interpolated"
    ]
    if len(preferred_sources) >= max(5, len(valid) // 3):
        valid = preferred_sources
    if not occlusion_filter:
        return valid
    filtered = [
        idx
        for idx in valid
        if (
            records[idx].quality_score >= 0.18
            and records[idx].occlusion_score <= 0.72
            and records[idx].signal_ratio >= 0.006
            and records[idx].mouth_likeness_score >= 0.10
        )
    ]
    return filtered if len(filtered) >= max(3, min(8, len(valid) // 3)) else valid


def _select_sprite_frames(records: list[MouthRecord], indices: list[int] | None = None) -> dict[str, int]:
    source_indices = indices or list(range(len(records)))
    if not source_indices:
        source_indices = list(range(len(records)))
    subset = [records[idx] for idx in source_indices]
    ratios = np.asarray([rec.open_ratio for rec in subset], dtype=np.float32)
    widths = np.asarray([rec.width for rec in subset], dtype=np.float32)
    heights = np.asarray([rec.height for rec in subset], dtype=np.float32)
    quality = np.asarray([max(0.05, rec.quality_score) for rec in subset], dtype=np.float32)
    closed = int(np.argmin(ratios + (1.0 - quality) * 0.08))
    open_idx = int(np.argmax(ratios + quality * 0.05))
    mid = float(np.percentile(ratios, 55))
    half = int(np.argmin(np.abs(ratios - mid) + (1.0 - quality) * 0.05))
    wide_score = widths / np.maximum(heights, 1.0)
    e_idx = int(np.argmax(wide_score - ratios * 0.5 + quality * 0.08))
    round_score = ratios - (widths / np.maximum(widths.max(), 1.0)) * 0.2 + quality * 0.05
    u_idx = int(np.argmax(round_score))
    return {
        "closed": source_indices[closed],
        "half": source_indices[half],
        "open": source_indices[open_idx],
        "e": source_indices[e_idx],
        "u": source_indices[u_idx],
    }


def _quad_angle_degrees(quad: list[list[float]]) -> float:
    dx = float(quad[1][0] - quad[0][0])
    dy = float(quad[1][1] - quad[0][1])
    return math.degrees(math.atan2(dy, dx))


def _normalize_angle_params(angle_range_degrees: int, angle_step_degrees: int) -> tuple[int, int]:
    step = max(5, min(30, int(angle_step_degrees)))
    step = max(5, int(round(step / 5.0) * 5))
    angle_range = max(step, min(75, int(angle_range_degrees)))
    angle_range = int(round(angle_range / float(step)) * step)
    return angle_range, step


def _angle_bins(angle_range_degrees: int, angle_step_degrees: int) -> list[tuple[int, str]]:
    angle_range, step = _normalize_angle_params(angle_range_degrees, angle_step_degrees)
    return [_angle_bin_label(value, angle_range_degrees=angle_range, angle_step_degrees=step) for value in range(-angle_range, angle_range + 1, step)]


def _angle_bin_label(
    angle_degrees: float,
    *,
    angle_range_degrees: int = DEFAULT_ANGLE_RANGE_DEGREES,
    angle_step_degrees: int = DEFAULT_ANGLE_STEP_DEGREES,
) -> tuple[int, str]:
    angle_range, step = _normalize_angle_params(angle_range_degrees, angle_step_degrees)
    value = int(round(angle_degrees / float(step)) * step)
    value = max(-angle_range, min(angle_range, value))
    prefix = "p" if value >= 0 else "m"
    return value, f"angle_{prefix}{abs(value):02d}"


def _angle_degrees_from_label(label: str) -> int:
    match = re.fullmatch(r"angle_([pm])(\d+)", label)
    if not match:
        return 0
    value = int(match.group(2))
    return value if match.group(1) == "p" else -value


def _select_angle_sprite_frames(
    records: list[MouthRecord],
    *,
    angle_range_degrees: int,
    angle_step_degrees: int,
    occlusion_filter: bool,
) -> dict[str, dict[str, int]]:
    eligible = set(_eligible_record_indices(records, occlusion_filter=occlusion_filter))
    groups: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        if idx not in eligible or not rec.valid or not rec.track_quad:
            continue
        _, label = _angle_bin_label(
            _quad_angle_degrees(rec.track_quad),
            angle_range_degrees=angle_range_degrees,
            angle_step_degrees=angle_step_degrees,
        )
        groups.setdefault(label, []).append(idx)

    selected: dict[str, dict[str, int]] = {}
    for label, indices in sorted(groups.items()):
        if not indices:
            continue
        selected[label] = _select_sprite_frames(records, indices)
    return selected


def _shape_open_thresholds(records: list[MouthRecord], indices: list[int]) -> dict[str, tuple[float, float]]:
    if not indices:
        values = np.asarray([rec.open_ratio for rec in records], dtype=np.float32)
    else:
        values = np.asarray([records[idx].open_ratio for idx in indices], dtype=np.float32)
    p10 = float(np.percentile(values, 10))
    p35 = float(np.percentile(values, 35))
    p65 = float(np.percentile(values, 65))
    p90 = float(np.percentile(values, 90))
    return {
        "closed": (-1.0, p10 + max(0.035, (p35 - p10) * 0.75)),
        "half": (p35 - 0.08, p65 + 0.08),
        "open": (p90 - max(0.075, (p90 - p65) * 0.75), 2.0),
        "e": (p35 - 0.10, p90 + 0.10),
        "u": (p65 - 0.10, 2.0),
    }


def _slot_matches_open_band(records: list[MouthRecord], frame_idx: int, slot: str, thresholds: dict[str, tuple[float, float]]) -> bool:
    low, high = thresholds.get(slot, (-1.0, 2.0))
    rec = records[frame_idx]
    return low <= rec.open_ratio <= high and rec.quality_score >= 0.78 and rec.mouth_likeness_score >= 0.50


def _central_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask
    h, w = mask.shape
    center = np.array([w * 0.5, h * 0.5])
    best_label = 1
    best_score = -1e18
    for label in range(1, num):
        area = float(stats[label, cv2.CC_STAT_AREA])
        dist = float(np.linalg.norm(centroids[label] - center))
        score = area - dist * 2.0
        if score > best_score:
            best_score = score
            best_label = label
    return labels == best_label


def _scaled_bbox(
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0, 8.0) * scale
    sx0 = max(0, int(round(cx - side * 0.5)))
    sy0 = max(0, int(round(cy - side * 0.5)))
    sx1 = min(width - 1, int(round(cx + side * 0.5)))
    sy1 = min(height - 1, int(round(cy + side * 0.5)))
    return sx0, sy0, sx1, sy1


def _make_sprite(
    frame_bgr: np.ndarray,
    mouth_bbox: tuple[float, float, float, float],
    sprite_size: int,
) -> Image.Image:
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _scaled_bbox(mouth_bbox, scale=1.45, width=w, height=h)
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGBA", (sprite_size, sprite_size), (0, 0, 0, 0))

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask, mouth_window = _mouth_signal_mask(crop_bgr, rx_scale=0.38, ry_scale=0.23)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    component_result = _best_mouth_component(mask, crop_bgr, mouth_window)
    if component_result is not None:
        mask = component_result[0]
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    if int(mask.sum()) < max(12, mask.size // 300):
        edges = cv2.Canny(gray, 80, 170) > 0
        dark = gray < 100
        mask = ((dark | edges) & mouth_window).astype(np.uint8)
        fallback_component = _best_mouth_component(mask, crop_bgr, mouth_window)
        if fallback_component is not None:
            mask = fallback_component[0].astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    mask = _central_component(mask)
    alpha = cv2.GaussianBlur((mask.astype(np.uint8) * 255), (5, 5), 0)
    rgba = np.dstack([crop_rgb, alpha])
    sprite = Image.fromarray(rgba, "RGBA")
    sprite.thumbnail((sprite_size, sprite_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (sprite_size, sprite_size), (0, 0, 0, 0))
    canvas.alpha_composite(sprite, ((sprite_size - sprite.width) // 2, (sprite_size - sprite.height) // 2))
    return canvas


def _inpaint_mask(
    shape: tuple[int, int],
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bw = max(4.0, x1 - x0) * scale
    bh = max(4.0, y1 - y0) * scale
    ix0 = max(0, int(round(cx - bw * 0.5)))
    iy0 = max(0, int(round(cy - bh * 0.5)))
    ix1 = min(width - 1, int(round(cx + bw * 0.5)))
    iy1 = min(height - 1, int(round(cy + bh * 0.5)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, ((ix0 + ix1) // 2, (iy0 + iy1) // 2), (max(2, (ix1 - ix0) // 2), max(2, (iy1 - iy0) // 2)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def _read_video_frame(source: Path, frame_idx: int, *, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if ok:
        return frame
    return np.zeros((height, width, 3), dtype=np.uint8)


def _generate_angle_sprite_file(source_path: Path, target_path: Path, angle_degrees: float) -> None:
    sprite = Image.open(source_path).convert("RGBA")
    alpha = sprite.getchannel("A")
    bbox = alpha.getbbox()
    center = (sprite.width * 0.5, sprite.height * 0.5)
    if bbox is not None:
        center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    rotated = sprite.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        center=center,
    )
    rotated.save(target_path, format="PNG", optimize=True)


def _first_rgb_image(image: Any) -> np.ndarray:
    data = image
    if isinstance(data, (list, tuple)):
        if not data:
            raise RuntimeError("Generated mouth image input is empty.")
        data = data[0]
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"Generated mouth image must be HWC or BHWC, got shape {arr.shape}.")
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise RuntimeError(f"Generated mouth image must have 3 or 4 channels, got shape {arr.shape}.")
    return arr


def _make_sprite_from_generated_edit(image: Any, sprite_size: int) -> Image.Image:
    rgb = _first_rgb_image(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask, mouth_window = _mouth_signal_mask(bgr, rx_scale=0.30, ry_scale=0.24)
    component_result = _best_mouth_component(mask.astype(np.uint8), bgr, mouth_window)
    if component_result is not None:
        mask = component_result[0].astype(np.uint8)
    else:
        mask = (mask & mouth_window).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    ys, xs = np.where(mask > 0)
    canvas = Image.new("RGBA", (sprite_size, sprite_size), (0, 0, 0, 0))
    if len(xs) == 0:
        return canvas

    x0 = max(0, int(xs.min()) - 10)
    x1 = min(rgb.shape[1], int(xs.max()) + 11)
    y0 = max(0, int(ys.min()) - 10)
    y1 = min(rgb.shape[0], int(ys.max()) + 11)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    alpha = cv2.GaussianBlur((crop_mask * 255).astype(np.uint8), (5, 5), 0)
    sprite = Image.fromarray(np.dstack([crop_rgb, alpha]), "RGBA")
    sprite.thumbnail((max(16, int(sprite_size * 0.43)), max(16, int(sprite_size * 0.43))), Image.Resampling.LANCZOS)
    canvas.alpha_composite(sprite, ((sprite_size - sprite.width) // 2, (sprite_size - sprite.height) // 2))
    return canvas


def _quality_summary(
    records: list[MouthRecord],
    angle_set_meta: dict[str, dict[str, Any]],
    *,
    angle_range_degrees: int,
    angle_step_degrees: int,
) -> dict[str, Any]:
    qualities = np.asarray([rec.quality_score for rec in records], dtype=np.float32)
    occlusions = np.asarray([rec.occlusion_score for rec in records], dtype=np.float32)
    angles = np.asarray([
        _quad_angle_degrees(rec.track_quad) if rec.track_quad else 0.0
        for rec in records
    ], dtype=np.float32)
    generated_sets = sorted(label for label, meta in angle_set_meta.items() if meta.get("generated"))
    real_sets = sorted(label for label, meta in angle_set_meta.items() if not meta.get("generated"))
    warnings: list[str] = []
    if float(qualities.mean()) < 0.24:
        warnings.append("low_average_mouth_signal")
    if float((occlusions > 0.72).mean()) > 0.20:
        warnings.append("many_occluded_or_low_signal_frames")
    if len(generated_sets) > len(real_sets):
        warnings.append("most_angle_sets_generated_from_nearest_real_set")
    if float(np.max(np.abs(angles))) >= angle_range_degrees - max(1, angle_step_degrees * 0.5):
        warnings.append("mouth_angle_hits_configured_range_edge")

    return {
        "frameQuality": {
            "average": round(float(qualities.mean()), 6),
            "min": round(float(qualities.min()), 6),
            "max": round(float(qualities.max()), 6),
        },
        "occlusion": {
            "average": round(float(occlusions.mean()), 6),
            "highFrameRatio": round(float((occlusions > 0.72).mean()), 6),
        },
        "angle": {
            "rangeDegrees": int(angle_range_degrees),
            "stepDegrees": int(angle_step_degrees),
            "minObserved": round(float(angles.min()), 4),
            "maxObserved": round(float(angles.max()), 4),
            "realSets": real_sets,
            "generatedSets": generated_sets,
        },
        "sourceFrames": {
            "interpolated": sum(1 for rec in records if rec.source == "interpolated"),
            "direct": sum(1 for rec in records if rec.source != "interpolated"),
        },
        "warnings": warnings,
    }


def _mouth_articulation_summary(records: list[MouthRecord]) -> dict[str, Any]:
    opens = np.asarray([rec.open_ratio for rec in records], dtype=np.float32)
    low = float(np.percentile(opens, 10))
    high = float(np.percentile(opens, 90))
    open_range = high - low
    requires_generation = open_range < MIN_EXTRACTED_OPEN_RANGE
    return {
        "p10": round(low, 6),
        "p90": round(high, 6),
        "range": round(open_range, 6),
        "minRequiredRange": MIN_EXTRACTED_OPEN_RANGE,
        "requiresModelGeneration": bool(requires_generation),
        "reason": "insufficient_mouth_open_variation" if requires_generation else None,
    }


def _write_mouth_generation_plan(
    bundle_dir: Path,
    source: Path,
    records: list[MouthRecord],
    sprite_indices: dict[str, int],
    *,
    width: int,
    height: int,
    articulation: dict[str, Any],
) -> dict[str, Any] | None:
    if not articulation.get("requiresModelGeneration"):
        return None

    ref_idx = sprite_indices.get("closed", 0)
    rec = records[ref_idx]
    if rec.mouth_bbox is None:
        return None

    frame = _read_video_frame(source, ref_idx, width=width, height=height)
    x0, y0, x1, y1 = _scaled_bbox(rec.mouth_bbox, scale=4.0, width=width, height=height)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    generation_dir = bundle_dir / "mouth_generation_inputs"
    generation_dir.mkdir(parents=True, exist_ok=True)
    reference_path = generation_dir / "reference_face_mouth_crop.png"
    mask_path = generation_dir / "mouth_edit_mask.png"
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(reference_path, format="PNG")

    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    bx0, by0, bx1, by1 = rec.mouth_bbox
    cx = int(round(((bx0 + bx1) * 0.5) - x0))
    cy = int(round(((by0 + by1) * 0.5) - y0))
    rx = max(4, int(round((bx1 - bx0) * 0.95)))
    ry = max(4, int(round((by1 - by0) * 1.35)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    Image.fromarray(mask).save(mask_path, format="PNG")

    plan = {
        "schema": "pngtuber.mouthGenerationPlan.v1",
        "required": True,
        "reason": articulation.get("reason"),
        "localOnly": True,
        "sourceFrame": ref_idx,
        "inputs": {
            "referenceFaceMouthCrop": str(reference_path),
            "mouthEditMask": str(mask_path),
        },
        "recommendedLocalModels": {
            "diffusionModel": "qwen_image_edit_2511_bf16_각도.safetensors",
            "textEncoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "vae": "qwen_image_vae.safetensors",
            "lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        },
        "shapePrompts": {
            "open": "anime character mouth only, same character, same lighting, natural open speaking mouth, preserve face and style",
            "half": "anime character mouth only, same character, same lighting, small half-open speaking mouth, preserve face and style",
            "e": "anime character mouth only, same character, same lighting, narrow smiling e vowel mouth, preserve face and style",
            "u": "anime character mouth only, same character, same lighting, rounded u vowel mouth, preserve face and style",
        },
        "negativePrompt": "different character, changed face, changed eyes, teeth horror, extra mouth, deformed lips, blurry, low quality",
        "outputContract": {
            "open": "mouth/open.png",
            "half": "mouth/half.png",
            "e": "mouth/e.png",
            "u": "mouth/u.png",
        },
    }
    plan_path = generation_dir / "mouth_generation_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"plan": str(plan_path), **plan["inputs"]}


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _mux_audio_and_h264(no_audio_path: Path, source_video: Path, final_path: Path, *, preserve_audio: bool) -> None:
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        shutil.move(no_audio_path, final_path)
        return
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(no_audio_path),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
    ]
    if preserve_audio:
        cmd += ["-map", "1:a?"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium"]
    if preserve_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(final_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        shutil.move(no_audio_path, final_path)
    else:
        no_audio_path.unlink(missing_ok=True)


class PNGTuberVideoMouthBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input_video.mp4", "multiline": False}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "asset_id": ("STRING", {"default": "pngtuber_video_mouth", "multiline": False}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 20000, "step": 1}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1}),
                "detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.95, "step": 0.05}),
                "detection_mode": (["anime_first", "mediapipe_first", "anime_only", "face_yolo_only"], {"default": "anime_first"}),
                "face_yolo_fallback": ("BOOLEAN", {"default": True}),
                "track_quad_scale": ("FLOAT", {"default": 1.8, "min": 1.2, "max": 8.0, "step": 0.1}),
                "inpaint_scale": ("FLOAT", {"default": 1.55, "min": 1.0, "max": 4.0, "step": 0.05}),
                "inpaint_radius": ("INT", {"default": 5, "min": 1, "max": 31, "step": 1}),
                "sprite_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "preserve_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "angle_range_degrees": ("INT", {"default": 45, "min": 15, "max": 75, "step": 15}),
                "angle_step_degrees": ("INT", {"default": 15, "min": 5, "max": 30, "step": 5}),
                "occlusion_filter": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "mouthless_video",
        "mouth_track_json",
        "mouth_sprite_atlas_json",
        "bundle_manifest_json",
        "mouth_closed",
        "mouth_half",
        "mouth_open",
        "mouth_e",
        "mouth_u",
        "summary_json",
    )
    FUNCTION = "run"
    CATEGORY = "PNGTuber/Video Mouth"
    OUTPUT_NODE = True

    def run(
        self,
        video_path: str,
        output_dir: str,
        asset_id: str,
        max_frames: int,
        frame_stride: int,
        detection_confidence: float,
        detection_mode: str,
        face_yolo_fallback: bool,
        track_quad_scale: float,
        inpaint_scale: float,
        inpaint_radius: int,
        sprite_size: int,
        preserve_audio: bool,
        angle_range_degrees: int = DEFAULT_ANGLE_RANGE_DEGREES,
        angle_step_degrees: int = DEFAULT_ANGLE_STEP_DEGREES,
        occlusion_filter: bool = True,
    ):
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError(
                "mediapipe is required. Install it with ComfyUI's venv pip: "
                f"{sys.executable} -m pip install -r custom_nodes/ComfyUI-PromptMaker-PNGTuber/requirements.txt"
            ) from exc

        source = _resolve_path(video_path)
        if not source.exists():
            raise RuntimeError(f"Video not found: {source}")

        asset = _safe_asset_id(asset_id)
        root = Path(output_dir).expanduser() if output_dir.strip() else _comfy_output_dir() / "pngtuber_video_mouth"
        bundle_dir = root / asset
        mouth_dir = bundle_dir / "mouth"
        mouth_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = total if max_frames <= 0 else min(total, max_frames)
        if width <= 0 or height <= 0 or limit <= 0:
            cap.release()
            raise RuntimeError("Video has no readable frames.")
        angle_range_degrees, angle_step_degrees = _normalize_angle_params(angle_range_degrees, angle_step_degrees)

        records: list[MouthRecord] = []
        anime_cascade = _load_anime_cascade()
        fallback_model = _load_face_yolo_model() if face_yolo_fallback else None
        face_mesh_cm = None
        face_mesh = None
        try:
            face_mesh_cm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=float(detection_confidence),
                min_tracking_confidence=float(detection_confidence),
            )
            face_mesh = face_mesh_cm.__enter__()
        except Exception:
            face_mesh = None
        try:
            idx = 0
            last_detection = MouthRecord(valid=False)
            while idx < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_stride == 0 or not last_detection.valid:
                    if detection_mode in ("anime_first", "anime_only"):
                        last_detection = _anime_cascade_mouth_fallback(
                            frame,
                            anime_cascade,
                            quad_scale=float(track_quad_scale),
                        )
                    else:
                        last_detection = MouthRecord(valid=False)
                    if not last_detection.valid and face_mesh is not None and detection_mode != "face_yolo_only":
                        try:
                            last_detection = _detect_mouth(
                                frame,
                                face_mesh,
                                min_confidence=float(detection_confidence),
                                quad_scale=float(track_quad_scale),
                            )
                        except Exception:
                            last_detection = MouthRecord(valid=False)
                    else:
                        if not last_detection.valid:
                            last_detection = MouthRecord(valid=False)
                    if not last_detection.valid:
                        last_detection = _face_yolo_mouth_fallback(
                            frame,
                            fallback_model,
                            quad_scale=float(track_quad_scale),
                        )
                    last_detection = _with_quality(last_detection, frame)
                    records.append(last_detection)
                else:
                    records.append(MouthRecord(valid=False))
                idx += 1
        finally:
            if face_mesh_cm is not None:
                face_mesh_cm.__exit__(None, None, None)
        cap.release()
        if not records:
            raise RuntimeError("Video has no readable frames.")

        records = _interpolate_records(records)
        eligible_indices = _eligible_record_indices(records, occlusion_filter=bool(occlusion_filter))
        shape_open_thresholds = _shape_open_thresholds(records, eligible_indices)
        sprite_indices = _select_sprite_frames(records, eligible_indices)
        angle_sprite_indices = _select_angle_sprite_frames(
            records,
            angle_range_degrees=angle_range_degrees,
            angle_step_degrees=angle_step_degrees,
            occlusion_filter=bool(occlusion_filter),
        )

        sprite_frames: dict[str, np.ndarray] = {}
        no_audio_path = bundle_dir / "_mouthless_noaudio.mp4"
        final_video_path = bundle_dir / "loop_mouthless_h264.mp4"
        writer = cv2.VideoWriter(
            str(no_audio_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {no_audio_path}")

        cap = cv2.VideoCapture(str(source))
        for idx, rec in enumerate(records):
            ok, frame = cap.read()
            if not ok:
                break
            for slot, frame_idx in sprite_indices.items():
                if idx == frame_idx:
                    sprite_frames[slot] = frame.copy()
            mask = _inpaint_mask((height, width), rec.mouth_bbox, scale=float(inpaint_scale))  # type: ignore[arg-type]
            inpainted = cv2.inpaint(frame, mask, int(inpaint_radius), cv2.INPAINT_TELEA)
            writer.write(inpainted)
        cap.release()
        writer.release()
        _mux_audio_and_h264(no_audio_path, source, final_video_path, preserve_audio=bool(preserve_audio))

        sprite_paths: dict[str, Path] = {}
        for slot in MOUTH_SHAPES:
            frame_idx = sprite_indices[slot]
            frame = sprite_frames.get(slot)
            if frame is None:
                frame = _read_video_frame(source, frame_idx, width=width, height=height)
            sprite = _make_sprite(frame, records[frame_idx].mouth_bbox, int(sprite_size))  # type: ignore[arg-type]
            out = mouth_dir / f"{slot}.png"
            sprite.save(out, format="PNG", optimize=True)
            sprite_paths[slot] = out

        angle_sprite_paths: dict[str, dict[str, Path]] = {}
        angle_set_meta: dict[str, dict[str, Any]] = {}
        angle_root = mouth_dir / "angles"
        for label, slot_indices in angle_sprite_indices.items():
            label_dir = angle_root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            angle_sprite_paths[label] = {}
            for slot in MOUTH_SHAPES:
                frame_idx = slot_indices[slot]
                out = label_dir / f"{slot}.png"
                if _slot_matches_open_band(records, frame_idx, slot, shape_open_thresholds):
                    frame = _read_video_frame(source, frame_idx, width=width, height=height)
                    sprite = _make_sprite(frame, records[frame_idx].mouth_bbox, int(sprite_size))  # type: ignore[arg-type]
                    sprite.save(out, format="PNG", optimize=True)
                else:
                    source_angle = _quad_angle_degrees(records[sprite_indices[slot]].track_quad)  # type: ignore[arg-type]
                    target_angle = _quad_angle_degrees(records[frame_idx].track_quad)  # type: ignore[arg-type]
                    _generate_angle_sprite_file(sprite_paths[slot], out, float(target_angle - source_angle))
                    slot_indices[slot] = sprite_indices[slot]
                angle_sprite_paths[label][slot] = out
            angle_set_meta[label] = {
                "angleDegrees": _angle_degrees_from_label(label),
                "generated": False,
                "frames": slot_indices,
                "sourceSet": None,
            }

        required_angle_sets = _angle_bins(angle_range_degrees, angle_step_degrees)
        if not angle_sprite_paths:
            angle_sprite_paths["angle_p00"] = sprite_paths.copy()
            angle_set_meta["angle_p00"] = {
                "angleDegrees": 0,
                "generated": False,
                "frames": sprite_indices,
                "sourceSet": None,
            }
        available_angles = {
            angle_set_meta[label]["angleDegrees"]: label
            for label in angle_sprite_paths
            if label in angle_set_meta
        }
        for target_angle, target_label in required_angle_sets:
            if target_label in angle_sprite_paths:
                continue
            nearest_angle = min(available_angles, key=lambda value: abs(value - target_angle))
            source_label = available_angles[nearest_angle]
            label_dir = angle_root / target_label
            label_dir.mkdir(parents=True, exist_ok=True)
            angle_sprite_paths[target_label] = {}
            for slot in MOUTH_SHAPES:
                out = label_dir / f"{slot}.png"
                _generate_angle_sprite_file(angle_sprite_paths[source_label][slot], out, float(target_angle - nearest_angle))
                angle_sprite_paths[target_label][slot] = out
            angle_set_meta[target_label] = {
                "angleDegrees": target_angle,
                "generated": True,
                "frames": None,
                "sourceSet": source_label,
            }

        atlas_quality = _quality_summary(
            records,
            angle_set_meta,
            angle_range_degrees=angle_range_degrees,
            angle_step_degrees=angle_step_degrees,
        )
        articulation = _mouth_articulation_summary(records)
        if articulation["requiresModelGeneration"]:
            atlas_quality.setdefault("warnings", []).append("insufficient_mouth_open_variation_requires_model_generation")
        mouth_generation_inputs = _write_mouth_generation_plan(
            bundle_dir,
            source,
            records,
            sprite_indices,
            width=width,
            height=height,
            articulation=articulation,
        )
        atlas_payload = {
            "schema": "pngtuber.mouthSpriteAtlas.v1",
            "defaultSet": "angle_p00" if "angle_p00" in angle_sprite_paths else next(iter(angle_sprite_paths), None),
            "shapeOrder": list(MOUTH_SHAPES),
            "angleStepDegrees": angle_step_degrees,
            "angleRangeDegrees": angle_range_degrees,
            "articulation": articulation,
            "quality": atlas_quality,
            "sets": {
                label: {
                    **angle_set_meta[label],
                    "files": {slot: str(path) for slot, path in paths.items()},
                }
                for label, paths in angle_sprite_paths.items()
            },
        }
        atlas_path = bundle_dir / "mouth_sprite_atlas.json"
        atlas_path.write_text(json.dumps(atlas_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        frames_payload = [
            {
                "valid": True,
                "quad": [[round(float(x), 4), round(float(y), 4)] for x, y in rec.track_quad],  # type: ignore[union-attr]
                "mouthOpen": round(float(rec.open_ratio), 6),
                "mouthAngleDegrees": round(_quad_angle_degrees(rec.track_quad), 4),  # type: ignore[arg-type]
                "spriteSet": _angle_bin_label(
                    _quad_angle_degrees(rec.track_quad),  # type: ignore[arg-type]
                    angle_range_degrees=angle_range_degrees,
                    angle_step_degrees=angle_step_degrees,
                )[1],
                "qualityScore": round(float(rec.quality_score), 6),
                "occlusionScore": round(float(rec.occlusion_score), 6),
                "signalRatio": round(float(rec.signal_ratio), 6),
                "mouthLikenessScore": round(float(rec.mouth_likeness_score), 6),
                "source": rec.source,
            }
            for rec in records
        ]
        track_payload = {
            "schema": "pngtuber.mouthTrack.v1",
            "fps": fps,
            "width": width,
            "height": height,
            "refSpriteSize": [int(sprite_size), int(sprite_size)],
            "calibration": {"offset": [0.0, 0.0], "scale": 1.0, "rotation": 0.0},
            "calibrationApplied": False,
            "frames": frames_payload,
            "generator": {
                "name": "ComfyUI-PromptMaker-PNGTuber",
                "version": 1,
                "sourceVideo": str(source),
                "frameStride": int(frame_stride),
                "trackQuadScale": float(track_quad_scale),
                "inpaintScale": float(inpaint_scale),
                "angleRangeDegrees": int(angle_range_degrees),
                "angleStepDegrees": int(angle_step_degrees),
                "occlusionFilter": bool(occlusion_filter),
            },
        }
        track_path = bundle_dir / "mouth_track.json"
        track_path.write_text(json.dumps(track_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest = {
            "schema": "pngtuber.videoMouthBundle.v1",
            "asset_id": asset,
            "source_video": str(source),
            "output_dir": str(bundle_dir),
            "fps": fps,
            "width": width,
            "height": height,
            "frames": len(records),
            "detected_frames": sum(1 for r in records if r.source != "interpolated"),
            "eligible_sprite_candidate_frames": len(eligible_indices),
            "sprite_frames": sprite_indices,
            "mouth_shapes": list(MOUTH_SHAPES),
            "articulation": articulation,
            "quality": atlas_quality,
            "compatibility": {
                "flatMouthSprites": True,
                "angleSpriteAtlas": True,
                "promptMakerVideoMouth": True,
                "requiresModelMouthGeneration": bool(articulation["requiresModelGeneration"]),
            },
            "files": {
                "video": str(final_video_path),
                "mouth_track": str(track_path),
                "mouth_sprite_atlas": str(atlas_path),
                **{f"mouth_{slot}": str(path) for slot, path in sprite_paths.items()},
                **({"mouth_generation_inputs": mouth_generation_inputs} if mouth_generation_inputs else {}),
                "angle_sets": {
                    label: {slot: str(path) for slot, path in paths.items()}
                    for label, paths in angle_sprite_paths.items()
                },
            },
        }
        manifest_path = bundle_dir / "bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            **manifest,
            "schema": "pngtuber.videoMouthSummary.v1",
            "bundle_manifest": str(manifest_path),
        }
        summary_path = bundle_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return (
            str(final_video_path),
            str(track_path),
            str(atlas_path),
            str(manifest_path),
            str(sprite_paths["closed"]),
            str(sprite_paths["half"]),
            str(sprite_paths["open"]),
            str(sprite_paths["e"]),
            str(sprite_paths["u"]),
            str(summary_path),
        )


class PNGTuberVideoUploadToMouthBundle(PNGTuberVideoMouthBuilder):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_input_video_files(),),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "asset_id": ("STRING", {"default": "", "multiline": False}),
                "quality_preset": (["balanced", "fast_preview", "full_quality"], {"default": "balanced"}),
                "preserve_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "angle_range_degrees": ("INT", {"default": 45, "min": 15, "max": 75, "step": 15}),
                "angle_step_degrees": ("INT", {"default": 15, "min": 5, "max": 30, "step": 5}),
                "advanced_video_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = PNGTuberVideoMouthBuilder.RETURN_TYPES
    RETURN_NAMES = PNGTuberVideoMouthBuilder.RETURN_NAMES
    FUNCTION = "run"
    CATEGORY = "PNGTuber/Video Mouth"
    OUTPUT_NODE = True

    def run(
        self,
        video: str,
        output_dir: str,
        asset_id: str,
        quality_preset: str,
        preserve_audio: bool,
        angle_range_degrees: int = DEFAULT_ANGLE_RANGE_DEGREES,
        angle_step_degrees: int = DEFAULT_ANGLE_STEP_DEGREES,
        advanced_video_path: str = "",
    ):
        video_value = advanced_video_path.strip() or video.strip()
        if not video_value:
            raise RuntimeError("Upload or select a source video before running the PNGTuber bundle workflow.")

        source = _resolve_path(video_value)
        asset = asset_id.strip() or _safe_asset_id(source.stem)
        presets = {
            "fast_preview": {"max_frames": 240, "frame_stride": 2, "sprite_size": 384},
            "balanced": {"max_frames": 0, "frame_stride": 1, "sprite_size": 512},
            "full_quality": {"max_frames": 0, "frame_stride": 1, "sprite_size": 768},
        }
        preset = presets.get(quality_preset, presets["balanced"])
        return super().run(
            video_path=str(source),
            output_dir=output_dir,
            asset_id=asset,
            max_frames=preset["max_frames"],
            frame_stride=preset["frame_stride"],
            detection_confidence=0.5,
            detection_mode="anime_first",
            face_yolo_fallback=True,
            track_quad_scale=1.8,
            inpaint_scale=1.55,
            inpaint_radius=5,
            sprite_size=preset["sprite_size"],
            preserve_audio=preserve_audio,
            angle_range_degrees=angle_range_degrees,
            angle_step_degrees=angle_step_degrees,
            occlusion_filter=True,
        )


class PNGTuberGeneratedMouthSpriteApplier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_manifest_path": ("STRING", {"default": ""}),
                "sprite_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "overwrite_existing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "closed_image": ("IMAGE",),
                "open_image": ("IMAGE",),
                "half_image": ("IMAGE",),
                "e_image": ("IMAGE",),
                "u_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("bundle_manifest", "mouth_sprite_atlas", "mouth_open", "mouth_half", "mouth_e", "mouth_u")
    FUNCTION = "run"
    CATEGORY = "PromptMaker/PNGTuber"

    def run(
        self,
        bundle_manifest_path: str,
        sprite_size: int,
        overwrite_existing: bool,
        closed_image: Any = None,
        open_image: Any = None,
        half_image: Any = None,
        e_image: Any = None,
        u_image: Any = None,
    ):
        manifest_path = _resolve_path(bundle_manifest_path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "bundle_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Bundle manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_dir = Path(manifest.get("output_dir") or manifest_path.parent)
        atlas_path = Path(manifest.get("files", {}).get("mouth_sprite_atlas") or bundle_dir / "mouth_sprite_atlas.json")
        if not atlas_path.exists():
            raise RuntimeError(f"Mouth sprite atlas not found: {atlas_path}")
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))

        shape_inputs = {
            "closed": closed_image,
            "open": open_image,
            "half": half_image,
            "e": e_image,
            "u": u_image,
        }
        generated_shapes = [shape for shape, image in shape_inputs.items() if image is not None]
        if not generated_shapes:
            raise RuntimeError("Provide at least one generated mouth image.")

        mouth_dir = bundle_dir / "mouth"
        angle_root = mouth_dir / "angles"
        mouth_dir.mkdir(parents=True, exist_ok=True)
        model_output_files: dict[str, str] = {}
        for shape in generated_shapes:
            flat_path = mouth_dir / f"{shape}.png"
            if flat_path.exists() and not overwrite_existing:
                model_output_files[shape] = str(flat_path)
                continue
            sprite = _make_sprite_from_generated_edit(shape_inputs[shape], int(sprite_size))
            sprite.save(flat_path, format="PNG", optimize=True)
            model_output_files[shape] = str(flat_path)

        sets = atlas.setdefault("sets", {})
        if "angle_p00" not in sets:
            sets["angle_p00"] = {
                "angleDegrees": 0,
                "generated": False,
                "frames": None,
                "sourceSet": None,
                "files": {},
            }
        for label, meta in sets.items():
            angle_degrees = int(meta.get("angleDegrees") or _angle_degrees_from_label(label))
            label_dir = angle_root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            files = meta.setdefault("files", {})
            for shape in generated_shapes:
                target = label_dir / f"{shape}.png"
                if angle_degrees == 0:
                    shutil.copyfile(model_output_files[shape], target)
                else:
                    _generate_angle_sprite_file(Path(model_output_files[shape]), target, float(angle_degrees))
                files[shape] = str(target)

        model_generation = atlas.setdefault("modelGeneration", {})
        completed = sorted(set(model_generation.get("completedShapes", [])) | set(generated_shapes))
        model_generation.update({
            "completed": True,
            "completedShapes": completed,
            "source": "PNGTuberGeneratedMouthSpriteApplier",
        })
        if atlas.get("articulation", {}).get("requiresModelGeneration"):
            atlas["articulation"]["requiresModelGeneration"] = False
            atlas["articulation"]["modelGenerationCompleted"] = True
        atlas_path.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), encoding="utf-8")

        files = manifest.setdefault("files", {})
        for shape, path in model_output_files.items():
            files[f"mouth_{shape}"] = path
        files["mouth_sprite_atlas"] = str(atlas_path)
        files["model_generation_outputs"] = model_output_files
        manifest.setdefault("compatibility", {})["requiresModelMouthGeneration"] = False
        manifest["modelGeneration"] = atlas["modelGeneration"]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_path = bundle_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(manifest)
            summary["schema"] = "pngtuber.videoMouthSummary.v1"
            summary["bundle_manifest"] = str(manifest_path)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return (
            str(manifest_path),
            str(atlas_path),
            files.get("mouth_open", ""),
            files.get("mouth_half", ""),
            files.get("mouth_e", ""),
            files.get("mouth_u", ""),
        )


PromptMakerPNGTuberVideoMouth = PNGTuberVideoMouthBuilder

NODE_CLASS_MAPPINGS = {
    "PNGTuberVideoMouthBuilder": PNGTuberVideoMouthBuilder,
    "PNGTuberVideoUploadToMouthBundle": PNGTuberVideoUploadToMouthBundle,
    "PNGTuberGeneratedMouthSpriteApplier": PNGTuberGeneratedMouthSpriteApplier,
    "PromptMakerPNGTuberVideoMouth": PromptMakerPNGTuberVideoMouth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PNGTuberVideoMouthBuilder": "PNGTuber Video Mouth Builder",
    "PNGTuberVideoUploadToMouthBundle": "PNGTuber Video Upload to Mouth Bundle",
    "PNGTuberGeneratedMouthSpriteApplier": "PNGTuber Generated Mouth Sprite Applier",
    "PromptMakerPNGTuberVideoMouth": "PromptMaker PNGTuber Video Mouth Pipeline (compat)",
}
