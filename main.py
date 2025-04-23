from Gaze_Tracking.gaze_tracking import GazeTracking
from Emotion_Detection.src.emotions import EmotionDetector
import cv2

emotion_detector = EmotionDetector(
    weights_path="Emotion_Detection/src/model.h5",
    haar_path="Emotion_Detection/src/haarcascade_frontalface_default.xml"
)

gaze = GazeTracking()

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Gaze Tracking ===
    gaze.refresh(frame)
    gaze_frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    elif gaze.is_top():
        text = "Looking up"
    elif gaze.is_bottom():
        text = "Looking down"

    cv2.putText(gaze_frame, f"Gaze: {text}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # === Emotion Detection ===
    emotion, confidence = emotion_detector.detect_emotion(frame)
    cv2.putText(gaze_frame, f"Emotion: {emotion} ({confidence*100:.2f}%)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display final result
    cv2.imshow("Emotion + Gaze Tracking", cv2.resize(gaze_frame, (1000, 700)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()