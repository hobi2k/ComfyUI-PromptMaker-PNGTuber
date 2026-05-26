from __future__ import annotations

import json
import math
import os
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


@dataclass
class MouthRecord:
    valid: bool
    mouth_bbox: tuple[float, float, float, float] | None = None
    track_quad: list[list[float]] | None = None
    open_ratio: float = 0.0
    width: float = 0.0
    height: float = 0.0
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


def _estimate_mouth_angle_from_frame(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> float:
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _scaled_bbox(bbox, scale=1.55, width=width, height=height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)
    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    cy, cx = crop.shape[0] * 0.5, crop.shape[1] * 0.5
    rx, ry = max(1.0, crop.shape[1] * 0.42), max(1.0, crop.shape[0] * 0.28)
    mouth_window = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    edges = cv2.Canny(gray, 80, 170) > 0
    dark = val < 105
    red_lip = ((hue < 12) | (hue > 164)) & (sat > 58) & (val < 235) & (val > 30)
    mask = (dark | red_lip | (edges & (val < 205))) & mouth_window
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
        return bbox, 0.08, _estimate_mouth_angle_from_frame(frame_bgr, bbox)

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
    open_ratio = float((y1 - y0) / max(1.0, x1 - x0))
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
            source=rec.source,
        )
        smoothed.append(prev)
    return smoothed


def _select_sprite_frames(records: list[MouthRecord]) -> dict[str, int]:
    ratios = np.asarray([rec.open_ratio for rec in records], dtype=np.float32)
    widths = np.asarray([rec.width for rec in records], dtype=np.float32)
    heights = np.asarray([rec.height for rec in records], dtype=np.float32)
    closed = int(np.argmin(ratios))
    open_idx = int(np.argmax(ratios))
    mid = float(np.percentile(ratios, 55))
    half = int(np.argmin(np.abs(ratios - mid)))
    wide_score = widths / np.maximum(heights, 1.0)
    e_idx = int(np.argmax(wide_score - ratios * 0.5))
    round_score = ratios - (widths / np.maximum(widths.max(), 1.0)) * 0.2
    u_idx = int(np.argmax(round_score))
    return {
        "closed": closed,
        "half": half,
        "open": open_idx,
        "e": e_idx,
        "u": u_idx,
    }


def _quad_angle_degrees(quad: list[list[float]]) -> float:
    dx = float(quad[1][0] - quad[0][0])
    dy = float(quad[1][1] - quad[0][1])
    return math.degrees(math.atan2(dy, dx))


def _angle_bin_label(angle_degrees: float) -> tuple[int, str]:
    value = int(round(angle_degrees / 15.0) * 15)
    value = max(-30, min(30, value))
    prefix = "p" if value >= 0 else "m"
    return value, f"angle_{prefix}{abs(value):02d}"


def _select_angle_sprite_frames(records: list[MouthRecord]) -> dict[str, dict[str, int]]:
    groups: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        if not rec.valid or not rec.track_quad:
            continue
        _, label = _angle_bin_label(_quad_angle_degrees(rec.track_quad))
        groups.setdefault(label, []).append(idx)

    selected: dict[str, dict[str, int]] = {}
    for label, indices in sorted(groups.items()):
        if not indices:
            continue
        subset = [records[idx] for idx in indices]
        local = _select_sprite_frames(subset)
        selected[label] = {slot: indices[local_idx] for slot, local_idx in local.items()}
    return selected


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
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    yy, xx = np.ogrid[:crop_bgr.shape[0], :crop_bgr.shape[1]]
    cy, cx = crop_bgr.shape[0] * 0.5, crop_bgr.shape[1] * 0.5
    rx, ry = max(1.0, crop_bgr.shape[1] * 0.38), max(1.0, crop_bgr.shape[0] * 0.23)
    mouth_window = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    edges = cv2.Canny(gray, 80, 170) > 0
    dark = val < 96
    red_lip = ((hue < 12) | (hue > 164)) & (sat > 62) & (val < 232) & (val > 35)
    mask = (dark | red_lip | (edges & (val < 205))) & mouth_window
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    if int(mask.sum()) < max(12, mask.size // 300):
        mask = ((dark | edges) & mouth_window).astype(np.uint8)
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


def _rotate_sprite_file(source_path: Path, target_path: Path, angle_degrees: float) -> None:
    sprite = Image.open(source_path).convert("RGBA")
    rotated = sprite.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        center=(sprite.width * 0.5, sprite.height * 0.5),
    )
    rotated.save(target_path, format="PNG", optimize=True)


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
        sprite_indices = _select_sprite_frames(records)
        angle_sprite_indices = _select_angle_sprite_frames(records)

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
        for slot in ("closed", "half", "open", "e", "u"):
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
            for slot in ("closed", "half", "open", "e", "u"):
                frame_idx = slot_indices[slot]
                frame = _read_video_frame(source, frame_idx, width=width, height=height)
                sprite = _make_sprite(frame, records[frame_idx].mouth_bbox, int(sprite_size))  # type: ignore[arg-type]
                out = label_dir / f"{slot}.png"
                sprite.save(out, format="PNG", optimize=True)
                angle_sprite_paths[label][slot] = out
            source_angle, _ = _angle_bin_label(_quad_angle_degrees(records[next(iter(slot_indices.values()))].track_quad))  # type: ignore[arg-type]
            angle_set_meta[label] = {
                "angleDegrees": source_angle,
                "generated": False,
                "frames": slot_indices,
                "sourceSet": None,
            }

        required_angle_sets = [(-30, "angle_m30"), (-15, "angle_m15"), (0, "angle_p00"), (15, "angle_p15"), (30, "angle_p30")]
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
            for slot in ("closed", "half", "open", "e", "u"):
                out = label_dir / f"{slot}.png"
                _rotate_sprite_file(angle_sprite_paths[source_label][slot], out, float(target_angle - nearest_angle))
                angle_sprite_paths[target_label][slot] = out
            angle_set_meta[target_label] = {
                "angleDegrees": target_angle,
                "generated": True,
                "frames": None,
                "sourceSet": source_label,
            }

        atlas_payload = {
            "schema": "pngtuber.mouthSpriteAtlas.v1",
            "defaultSet": "angle_p00" if "angle_p00" in angle_sprite_paths else next(iter(angle_sprite_paths), None),
            "shapeOrder": ["closed", "half", "open", "e", "u"],
            "angleStepDegrees": 15,
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
                "spriteSet": _angle_bin_label(_quad_angle_degrees(rec.track_quad))[1],  # type: ignore[arg-type]
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
            "detected_frames": sum(1 for r in records if r.source.startswith("mediapipe")),
            "sprite_frames": sprite_indices,
            "mouth_shapes": ["closed", "half", "open", "e", "u"],
            "compatibility": {
                "flatMouthSprites": True,
                "angleSpriteAtlas": True,
                "promptMakerVideoMouth": True,
            },
            "files": {
                "video": str(final_video_path),
                "mouth_track": str(track_path),
                "mouth_sprite_atlas": str(atlas_path),
                **{f"mouth_{slot}": str(path) for slot, path in sprite_paths.items()},
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


PromptMakerPNGTuberVideoMouth = PNGTuberVideoMouthBuilder

NODE_CLASS_MAPPINGS = {
    "PNGTuberVideoMouthBuilder": PNGTuberVideoMouthBuilder,
    "PromptMakerPNGTuberVideoMouth": PromptMakerPNGTuberVideoMouth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PNGTuberVideoMouthBuilder": "PNGTuber Video Mouth Builder",
    "PromptMakerPNGTuberVideoMouth": "PromptMaker PNGTuber Video Mouth Pipeline (compat)",
}
