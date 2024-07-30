import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

previous_area = 0

def calculate_orientation(hand_landmarks, frame_shape):
    """
    Calculate the orientation of the mobile device based on hand landmarks.
    """
    frame_w, frame_h = frame_shape[1], frame_shape[0]
    hand_center = np.mean([[lm.x * frame_w, lm.y * frame_h] for lm in hand_landmarks.landmark], axis=0)
    centerX, centerY = int(hand_center[0]), int(hand_center[1])
    
    horizontal_tilt = None
    vertical_tilt = None

    if centerX < frame_w // 3:
        horizontal_tilt = 'Left'
    elif centerX > 2 * frame_w // 3:
        horizontal_tilt = 'Right'

    if centerY < frame_h // 3:
        vertical_tilt = 'Up'
    elif centerY > 2 * frame_h // 3:
        vertical_tilt = 'Down'

    return horizontal_tilt, vertical_tilt

def calculate_bounding_box(hand_landmarks, frame_shape):
    """
    Calculate the bounding box of the detected hand.
    """
    frame_w, frame_h = frame_shape[1], frame_shape[0]
    x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame_w
    x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame_w
    y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame_h
    y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame_h
    return int(x_min), int(y_min), int(x_max), int(y_max)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate bounding box for the hand
            startX, startY, endX, endY = calculate_bounding_box(hand_landmarks, frame.shape)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            horizontal_tilt, vertical_tilt = calculate_orientation(hand_landmarks, frame.shape)
            
            if horizontal_tilt:
                cv2.putText(frame, f'Horizontal Tilt: {horizontal_tilt}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if vertical_tilt:
                cv2.putText(frame, f'Vertical Tilt: {vertical_tilt}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate movement towards or away from the camera
            current_area = (endX - startX) * (endY - startY)
            if previous_area != 0:
                if current_area > previous_area:
                    movement = "Moving Towards"
                else:
                    movement = "Moving Away"
                cv2.putText(frame, movement, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            previous_area = current_area

    else:
        cv2.putText(frame, "Device not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        previous_area = 0

    cv2.imshow('Mobile Device Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
