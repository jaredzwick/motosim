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
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """One detected cone."""
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in pixel coords
    center: Tuple[int, int]              # (cx, cy)
    area: int                            # pixel area
    aspect_ratio: float                  # h / w — cones should be > 1
    # Filled by project_to_world(); None until then.
    world_xy: Optional[Tuple[float, float]] = None


def project_pixel_to_ground(
    pixel_u: float,
    pixel_v: float,
    img_width: int,
    img_height: int,
    fovy_rad: float,
    cam_pos: np.ndarray,      # (3,) world xyz of camera
    cam_xmat: np.ndarray,     # (3, 3) world rotation of camera (MuJoCo convention)
    ground_z: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """
    Project an image pixel to world (x, y) on the ground plane z = ground_z.

    Uses ground-plane ray intersection: trace a ray from the camera center
    through the pixel into the world, find where it hits z=ground_z. Returns
    None if the ray points away from the ground (doesn't intersect).

    MuJoCo camera convention:
        camera -z axis is the look direction (into the scene)
        camera +y axis is up in image
        camera +x axis is right in image
        cam_xmat rows are the world-frame directions of local x, y, z axes.

    Args:
        pixel_u: pixel column (0 = left edge, img_width = right edge)
        pixel_v: pixel row    (0 = top edge,  img_height = bottom edge)
        img_width, img_height: render dimensions in pixels
        fovy_rad: vertical field of view in radians (= model.cam_fovy[cam_id] * pi/180)
        cam_pos: world position of camera (data.cam_xpos[cam_id])
        cam_xmat: 3x3 rotation (data.cam_xmat[cam_id].reshape(3, 3))

    Returns:
        (world_x, world_y) or None if no intersection with ground.
    """
    # Normalized device coords (NDC): [-1, +1] both axes, image y flipped
    ndc_x = (pixel_u - img_width  / 2.0) / (img_width  / 2.0)
    ndc_y = -(pixel_v - img_height / 2.0) / (img_height / 2.0)

    # Ray in camera local frame. Camera looks in -z.
    tan_half_fovy = np.tan(fovy_rad / 2.0)
    aspect        = img_width / img_height
    tan_half_fovx = tan_half_fovy * aspect

    ray_cam = np.array([
        ndc_x * tan_half_fovx,
        ndc_y * tan_half_fovy,
        -1.0,
    ])
    ray_cam /= np.linalg.norm(ray_cam)

    # Transform ray to world: world_dir = R @ cam_dir
    # cam_xmat columns are world-frame basis vectors for local x,y,z.
    ray_world = cam_xmat @ ray_cam

    # Intersect with z = ground_z plane: cam_pos + t * ray_world such that z = ground_z
    if abs(ray_world[2]) < 1e-6:
        return None  # ray parallel to ground
    t = (ground_z - cam_pos[2]) / ray_world[2]
    if t <= 0:
        return None  # intersection is behind camera

    world_x = cam_pos[0] + t * ray_world[0]
    world_y = cam_pos[1] + t * ray_world[1]
    return (float(world_x), float(world_y))


def project_detections_to_world(
    detections: List[Detection],
    img_width: int,
    img_height: int,
    fovy_rad: float,
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    cone_top_z: float = 0.65,
) -> None:
    """
    Project each detection to world (x, y), mutating Detection.world_xy.

    Projects the bounding-box TOP pixel against a horizontal plane at z=cone_top_z
    (cone apex height). The default 0.65 m matches the slalom cones in scene.xml
    (cylinder centered at z=0.35 with half-length 0.30, so top is at z=0.65).
    Using 0.70 undershoots systematically by ~5m because the camera sits HIGHER
    than the cone top (world z=0.8), so the ray through the bbox-top pixel goes
    downward and intersects any too-high plane closer to the camera. This is MUCH
    more accurate than projecting bbox-bottom against the ground plane, because
    the HSV detector reliably catches the cone's bright orange top but often
    misses the small base at long range. A bbox-bottom pixel that corresponds
    to the cone BODY'S bottom (not the ground) inverse-projects to a nearly
    horizontal ray that lands dozens of meters past the actual cone — which is
    exactly what made the bike drive through dead-ahead cones.
    """
    for det in detections:
        x, y, w, h = det.bbox
        pu = x + w / 2.0
        pv = y  # bbox top
        det.world_xy = project_pixel_to_ground(
            pu, pv, img_width, img_height, fovy_rad, cam_pos, cam_xmat,
            ground_z=cone_top_z,
        )


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
