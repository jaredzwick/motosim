"""
motosim — classical HSV-based cone detector.

This is the v0.4 perception module. Detects orange slalom cones in an RGB
frame using color thresholding and connected-component analysis. Zero ML
dependencies — scipy for connected components, OpenCV for HSV conversion
and image drawing.

Why classical instead of YOLO: orange cones on a grey asphalt plane is the
textbook best-case scenario for color-threshold CV. A pretrained YOLOv10
won't recognize them as "cones" (sim textures look nothing like photographic
training data), and training a custom detector on sim frames is a v0.4.1
upgrade. For the v0.4 goal of "the bike can see cones," a 30-line HSV
detector beats a 6MB neural net.

Upgrade path to YOLO:
    1. Save fpv frames from a drive rollout
    2. Label cones (or use RANSAC + known cone positions from scene.xml)
    3. Fine-tune ultralytics/yolov10n on the labeled set
    4. Replace detect_cones_hsv() with a YOLO inference call

Usage:
    from perception import detect_cones_hsv, annotate_frame
    cones = detect_cones_hsv(rgb_frame)
    annotated = annotate_frame(rgb_frame, cones)

Dependencies:
    pip install opencv-python
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    """One detected cone."""
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in pixel coords
    center: Tuple[int, int]              # (cx, cy)
    area: int                            # pixel area
    aspect_ratio: float                  # h / w — cones should be > 1


def detect_cones_hsv(rgb_frame: np.ndarray,
                     min_area: int = 40,
                     max_area: int = 20000,
                     min_aspect_ratio: float = 1.1) -> List[Detection]:
    """
    Detect orange cones in an RGB frame.

    Pipeline:
        1. Convert RGB -> HSV
        2. Threshold on orange hue band (H ~ 5-20) with high S, V
        3. Morphological close to merge disconnected cone parts (base + stripe + body)
        4. Connected-component labeling
        5. Filter by area and height-to-width aspect ratio (cones are tall)

    Args:
        rgb_frame: (H, W, 3) uint8 numpy array, RGB channel order
        min_area: minimum connected-component pixel area to consider a cone
        max_area: maximum area (rejects huge orange blobs like a sunset)
        min_aspect_ratio: minimum height / width (cones are vertically oriented)

    Returns:
        List of Detection objects, sorted left-to-right by center x.
    """
    import cv2  # import here so the module can be imported without opencv if someone only wants the types

    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Orange cone hue range. OpenCV H is in [0, 180].
    # Our cone_mat is rgba="0.95 0.35 0.08" which maps to H around 12.
    # Widen tolerance for shading variation.
    lower = np.array([2,  120, 120], dtype=np.uint8)
    upper = np.array([22, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Close small gaps between the cone's base + body + stripe segments.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0:
            continue
        ar = h / float(w)
        if ar < min_aspect_ratio:
            continue
        detections.append(Detection(
            bbox=(x, y, w, h),
            center=(x + w // 2, y + h // 2),
            area=area,
            aspect_ratio=round(ar, 2),
        ))

    detections.sort(key=lambda d: d.center[0])
    return detections


def annotate_frame(rgb_frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Return a copy of the frame with detection boxes + labels drawn.

    Box color: green. Label: "cone N (AR=2.3)".
    """
    import cv2

    out = rgb_frame.copy()
    for i, d in enumerate(detections):
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"cone {i + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(out, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Center dot
        cv2.circle(out, d.center, 3, (0, 0, 255), -1)

    # HUD text
    cv2.putText(out, f"cones detected: {len(detections)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
