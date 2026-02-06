import math
import cv2
import numpy as np
import mediapipe as mp

def landmarker_options(running_mode):
    match running_mode:
        case "Video":
            mode = VisionRunningMode.VIDEO
        case "Image":
            mode = VisionRunningMode.IMAGE
        case _:
            mode = VisionRunningMode.IMAGE
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode= mode
    )
    return options

def get_anchor_offset(width, height, point_name):
    """Maps an anchor string to specific pixel coordinates within the asset."""
    multipliers = {
        "top_left":      (0, 0),    "top_middle":    (0.5, 0), "top_right":     (1, 0),
        "middle_left":   (0, 0.5),  "middle":        (0.5, 0.5), "middle_right":  (1, 0.5),
        "bottom_left":   (0, 1),    "bottom_middle": (0.5, 1), "bottom_right":  (1, 1)
    }
    # Default to center if the name is misspelled
    mult_x, mult_y = multipliers.get(point_name, (0.5, 0.5))
    return int(width * mult_x), int(height * mult_y)


def blend_onto_frame(background, overlay_part, x, y):
    """Handles pixel-perfect placement and alpha transparency."""
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay_part.shape[:2]

    # Find the intersection between the screen and the asset
    screen_x1, screen_y1 = max(x, 0), max(y, 0)
    screen_x2, screen_y2 = min(x + ov_w, bg_w), min(y + ov_h, bg_h)

    # Calculate where to cut the asset
    asset_x1, asset_y1 = max(0, -x), max(0, -y)
    asset_x2 = asset_x1 + (screen_x2 - screen_x1)
    asset_y2 = asset_y1 + (screen_y2 - screen_y1)

    if screen_x1 < screen_x2 and screen_y1 < screen_y2:
        roi = background[screen_y1:screen_y2, screen_x1:screen_x2]
        asset_slice = overlay_part[asset_y1:asset_y2, asset_x1:asset_x2]

        # Apply alpha blending if the image has 4 channels
        if asset_slice.shape[2] == 4:
            alpha = asset_slice[:, :, 3:4] / 255.0
            color = asset_slice[:, :, :3]
            background[screen_y1:screen_y2, screen_x1:screen_x2] = (alpha * color + (1 - alpha) * roi).astype(np.uint8)
        else:
            background[screen_y1:screen_y2, screen_x1:screen_x2] = asset_slice[:, :, :3]

    return background


def overlay(image, asset, face_landmarks, config):
    if asset is None or face_landmarks is None:
        return image

    height, width = image.shape[:2]

    # 1. Calculate Face Geometry
    left_eye, right_eye = face_landmarks[33], face_landmarks[263]
    anchor_landmark = face_landmarks[config.index]

    target_anchor_px = (int(anchor_landmark.x * width), int(anchor_landmark.y * height))

    # Calculate tilt and size based on eye distance
    delta_x = (right_eye.x - left_eye.x) * width
    delta_y = (right_eye.y - left_eye.y) * height
    eye_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    tilt_angle = math.degrees(math.atan2(delta_y, delta_x))

    # 2. Prepare Asset
    target_w = int(eye_distance * config.ratio)
    target_h = int(target_w * (asset.shape[0] / asset.shape[1]))
    resized_asset = cv2.resize(asset, (target_w, target_h))

    # 3. Create Rotation Matrix
    center = (target_w // 2, target_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -tilt_angle, 1.0)

    # Expand boundaries so the corners don't get cut off during rotation
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    bound_w = int((target_h * sin) + (target_w * cos))
    bound_h = int((target_h * cos) + (target_w * sin))

    rotation_matrix[0, 2] += (bound_w / 2) - center[0]
    rotation_matrix[1, 2] += (bound_h / 2) - center[1]

    rotated_asset = cv2.warpAffine(resized_asset, rotation_matrix, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT)

    # 4. Final Alignment
    # Find the local anchor point within the asset and rotate it
    local_x, local_y = get_anchor_offset(target_w, target_h, config.point)
    rotated_anchor = rotation_matrix @ np.array([local_x, local_y, 1])

    # Calculate top-left corner for placement
    top_left_x = target_anchor_px[0] - int(rotated_anchor[0])
    top_left_y = target_anchor_px[1] - int(rotated_anchor[1])

    return blend_onto_frame(image, rotated_asset, top_left_x, top_left_y)

def get_ear(landmarks, eye_indices, w, h):
    # Map indices to (x, y) coordinates
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    # Vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # Horizontal distance
    h_dist = math.dist(pts[0], pts[3])

    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hat = cv2.imread("../assets/thug_life_hat.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.imread("../assets/thug_life_glasses.png", cv2.IMREAD_UNCHANGED)
blunt = cv2.imread("../assets/thug_life_blunt.png", cv2.IMREAD_UNCHANGED)
model_path: str = "../models/face_landmarker.task"