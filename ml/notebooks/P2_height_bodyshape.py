import os
import cv2
from ml.src.pose import PoseDetector
from ml.src.anthropometry import (
    calibrate_from_reference_object,
    estimate_height,
    classify_body_shape,
)

clicked_points = []
DEBUG_DIR = "ml/notebooks/debug_output"


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point captured: ({x}, {y})")


def capture_frame():
    cap = cv2.VideoCapture(0)
    frame = None
    print("Press 'c' to capture a frame, 'q' to quit without capturing.")
    while True:
        ok, live_frame = cap.read()
        if not ok:
            break
        cv2.imshow("P2 - press 'c' to capture", live_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            frame = live_frame.copy()
            break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyWindow("P2 - press 'c' to capture")
    return frame


def get_reference_pixel_height(frame):
    global clicked_points
    clicked_points = []

    window_name = "Click TOP then BOTTOM of reference object (place it near your FEET, same depth as you)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Click the TOP of your reference object, then its BOTTOM.")
    while len(clicked_points) < 2:
        display = frame.copy()
        for pt in clicked_points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)

    if len(clicked_points) < 2:
        return None

    (x1, y1), (x2, y2) = clicked_points
    pixel_height = abs(y2 - y1)
    return pixel_height, clicked_points


def save_debug_image(frame, pose_result, ref_points, head_top_y=None):
    """Saves the frame with landmarks, reference-object points, and the
    extrapolated head-top line drawn on it, so we can visually verify
    everything landed where it should."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_frame = frame.copy()
    h, w = debug_frame.shape[:2]

    # Draw all detected pose landmarks (green)
    for idx, lm in pose_result.landmarks.items():
        if lm.visibility < 0.5:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(debug_frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(debug_frame, str(idx), (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw reference object click points + line (red)
    if ref_points and len(ref_points) == 2:
        cv2.circle(debug_frame, ref_points[0], 6, (0, 0, 255), -1)
        cv2.circle(debug_frame, ref_points[1], 6, (0, 0, 255), -1)
        cv2.line(debug_frame, ref_points[0], ref_points[1], (0, 0, 255), 2)

    # Draw extrapolated head-top line (blue)
    if head_top_y is not None:
        y_px = int(head_top_y * h)
        cv2.line(debug_frame, (0, y_px), (w, y_px), (255, 0, 0), 1)

    out_path = os.path.join(DEBUG_DIR, "P2_debug.png")
    cv2.imwrite(out_path, debug_frame)
    print(f"Debug image saved to: {out_path}")


def main():
    frame = capture_frame()
    if frame is None:
        print("No frame captured. Exiting.")
        return

    ref_result = get_reference_pixel_height(frame)
    if ref_result is None:
        print("Reference object measurement failed. Exiting.")
        return
    pixel_height, ref_points = ref_result

    if pixel_height == 0:
        print("Reference object measurement failed (zero height). Exiting.")
        return

    real_height_cm = float(input("Enter the reference object's real height in cm: "))

    calibration = calibrate_from_reference_object(pixel_height, real_height_cm)
    print(f"Calibration: {calibration.pixel_per_cm:.2f} pixels/cm")

    with PoseDetector(static_image_mode=True) as detector:
        pose_result = detector.process(frame)

    if not pose_result.detected:
        print("No person detected in the captured frame. Exiting.")
        return

    # Recompute head_top_y here just for the debug drawing
    from ml.src.anthropometry.height_estimator import _estimate_head_top_y
    head_top_y = _estimate_head_top_y(pose_result.landmarks)

    height_result = estimate_height(pose_result, calibration)
    if height_result is None:
        print("Could not estimate height — required landmarks not visible enough.")
    else:
        print(f"Estimated height: {height_result.height_cm:.1f} cm "
              f"(confidence: {height_result.confidence:.2f})")

    shape_result = classify_body_shape(pose_result)
    print(f"Body shape: {shape_result.shape.value} "
          f"(shoulder/hip ratio: {shape_result.shoulder_hip_ratio}, "
          f"confidence: {shape_result.confidence:.2f})")

    save_debug_image(frame, pose_result, ref_points, head_top_y)


if __name__ == "__main__":
    main()