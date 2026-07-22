import numpy as np

VALID_ACTIONS = {
    "AZIMUTH_LEFT_COARSE",
    "AZIMUTH_RIGHT_COARSE",
    "AZIMUTH_LEFT_MEDIUM",
    "AZIMUTH_RIGHT_MEDIUM",
    "AZIMUTH_LEFT_FINE",
    "AZIMUTH_RIGHT_FINE",
    "AZIMUTH_RIGHT_180",
    "ELEVATION_UP_COARSE",
    "ELEVATION_DOWN_COARSE",
    "ELEVATION_UP_MEDIUM",
    "ELEVATION_DOWN_MEDIUM",
    "ELEVATION_UP_FINE",
    "ELEVATION_DOWN_FINE",
    "ELEVATION_UP_180",
    "ROLL_CW_90",
    "ROLL_CCW_90",
    "ROLL_CW_FINE",
    "ROLL_CCW_FINE",
    "ZOOM_IN",
    "ZOOM_OUT",
    "PAN_LEFT",
    "PAN_RIGHT",
    "PAN_UP",
    "PAN_DOWN",
    "UNDO_LAST",
    "STOP",
}

ACTION_DESCRIPTIONS = {
    "AZIMUTH_LEFT_COARSE":   "orbit camera 90 degrees left",
    "AZIMUTH_RIGHT_COARSE":  "orbit camera 90 degrees right",
    "AZIMUTH_LEFT_MEDIUM":   "orbit camera 45 degrees left",
    "AZIMUTH_RIGHT_MEDIUM":  "orbit camera 45 degrees right",
    "AZIMUTH_LEFT_FINE":     "orbit camera 15 degrees left",
    "AZIMUTH_RIGHT_FINE":    "orbit camera 15 degrees right",
    "AZIMUTH_RIGHT_180":     "orbit camera 180 degrees right (to the opposite side)",
    "ELEVATION_UP_COARSE":   "orbit camera 90 degrees upward",
    "ELEVATION_DOWN_COARSE": "orbit camera 90 degrees downward",
    "ELEVATION_UP_MEDIUM":   "orbit camera 45 degrees upward",
    "ELEVATION_DOWN_MEDIUM": "orbit camera 45 degrees downward",
    "ELEVATION_UP_FINE":     "orbit camera 15 degrees upward",
    "ELEVATION_DOWN_FINE":   "orbit camera 15 degrees downward",
    "ELEVATION_UP_180":      "orbit camera 180 degrees upward (to the opposite side)",
    "ROLL_CW_90":            "rotate image clockwise by 90 degrees",
    "ROLL_CCW_90":           "rotate image counterclockwise by 90 degrees",
    "ROLL_CW_FINE":          "rotate image clockwise by 5 degrees",
    "ROLL_CCW_FINE":         "rotate image counterclockwise by 5 degrees",
    "ZOOM_IN":               "make object larger",
    "ZOOM_OUT":              "make object smaller",
    "PAN_LEFT":              "move object left in the image",
    "PAN_RIGHT":             "move object right in the image",
    "PAN_UP":                "move object up in the image",
    "PAN_DOWN":              "move object down in the image",
    "UNDO_LAST":             "restore the previous camera state",
    "STOP":                  "use only when the current view matches the target",
}


def apply_action(action: str, camera, renderer):
    """Apply a named camera action. Returns True if the scene needs re-rendering."""
    if action == "STOP":
        return False

    if action == "AZIMUTH_LEFT_COARSE":
        camera.Azimuth(-90)
    elif action == "AZIMUTH_RIGHT_COARSE":
        camera.Azimuth(90)
    elif action == "AZIMUTH_LEFT_MEDIUM":
        camera.Azimuth(-45)
    elif action == "AZIMUTH_RIGHT_MEDIUM":
        camera.Azimuth(45)
    elif action == "AZIMUTH_LEFT_FINE":
        camera.Azimuth(-15)
    elif action == "AZIMUTH_RIGHT_FINE":
        camera.Azimuth(15)
    elif action == "AZIMUTH_RIGHT_180":
        camera.Azimuth(180)
    elif action == "ELEVATION_UP_COARSE":
        camera.Elevation(90)
    elif action == "ELEVATION_DOWN_COARSE":
        camera.Elevation(-90)
    elif action == "ELEVATION_UP_MEDIUM":
        camera.Elevation(45)
    elif action == "ELEVATION_DOWN_MEDIUM":
        camera.Elevation(-45)
    elif action == "ELEVATION_UP_FINE":
        camera.Elevation(15)
    elif action == "ELEVATION_DOWN_FINE":
        camera.Elevation(-15)
    elif action == "ELEVATION_UP_180":
        camera.Elevation(180)
    elif action == "ROLL_CW_90":
        camera.Roll(90)
    elif action == "ROLL_CCW_90":
        camera.Roll(-90)
    elif action == "ROLL_CW_FINE":
        camera.Roll(5)
    elif action == "ROLL_CCW_FINE":
        camera.Roll(-5)
    elif action == "ZOOM_IN":
        camera.Dolly(1.15)
    elif action == "ZOOM_OUT":
        camera.Dolly(0.87)
    elif action in ("PAN_LEFT", "PAN_RIGHT", "PAN_UP", "PAN_DOWN"):
        _apply_pan(action, camera)
    else:
        raise ValueError(f"Unknown action: {action}")

    camera.OrthogonalizeViewUp()
    renderer.ResetCameraClippingRange()
    return True


def _apply_pan(action: str, camera):
    pos = np.array(camera.GetPosition())
    fp = np.array(camera.GetFocalPoint())
    up = np.array(camera.GetViewUp())

    forward = fp - pos
    dist = np.linalg.norm(forward)
    step = 0.05 * dist

    forward_n = forward / dist
    right = np.cross(forward_n, up)
    right_n = right / np.linalg.norm(right)
    up_n = np.cross(right_n, forward_n)

    if action == "PAN_LEFT":
        delta = -step * right_n
    elif action == "PAN_RIGHT":
        delta = step * right_n
    elif action == "PAN_UP":
        delta = step * up_n
    elif action == "PAN_DOWN":
        delta = -step * up_n

    camera.SetPosition(*(pos + delta))
    camera.SetFocalPoint(*(fp + delta))
