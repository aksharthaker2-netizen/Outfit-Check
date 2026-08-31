import cv2
from ml.src.pose import PoseDetector, check_framing

def draw_landmarks(frame, pose_result):
    h, w = frame.shape[:2]
    for lm in pose_result.landmarks.values():
        if lm.visibility < 0.5:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture(0)
    with PoseDetector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.process(frame)
            status = check_framing(result)
            draw_landmarks(frame, result)
            color = (0, 200, 0) if status.is_valid else (0, 0, 255)
            cv2.putText(frame, status.guidance, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Outfit Check — P1 Pose Framing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()