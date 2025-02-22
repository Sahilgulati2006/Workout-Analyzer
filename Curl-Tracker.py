import cv2
import mediapipe as mp
import numpy as np
import time

exercises = {
    "bicep_curl": {
        "joints": ["shoulder", "elbow", "wrist"],
        "min_angle": 40,
        "max_angle": 160,
    },
    "squat": {"joints": ["hip", "knee", "ankle"], "min_angle": 160, "max_angle": 75},
    "push_up": {
        "joints": ["shoulder", "elbow", "wrist"],
        "min_angle": 70,
        "max_angle": 160,
    },
}


def calculate_angle(a, b, c):
    a = np.array(a)  # First joint
    b = np.array(b)  # Middle joint
    c = np.array(c)  # End joint

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def main():
    # Ask user for exercise selection
    exercise_name = (
        input("Enter exercise (bicep_curl, squat, push_up): ").strip().lower()
    )
    if exercise_name not in exercises:
        print("Invalid exercise. Defaulting to bicep_curl.")
        exercise_name = "bicep_curl"
    exercise = exercises[exercise_name]

    mp_pose = mp.solutions.pose

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error! Unable to access webcam.")
        exit()

    # Exercise Tracking Variables
    counter = 0
    # stage = "down"
    start_time = time.time()
    rep_times = []
    fatigue_threshold = 5  # Seconds per rep indicating fatigue

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract joint coordinates dynamically
                joint_names = exercise["joints"]
                joint_coords = []
                try:
                    for joint in joint_names:
                        landmark_index = getattr(
                            mp_pose.PoseLandmark, f"LEFT_{joint.upper()}"
                        ).value
                        joint_coords.append(
                            [landmarks[landmark_index].x, landmarks[landmark_index].y]
                        )
                except AttributeError:
                    pass
                if len(joint_coords) >= 3:
                    cv2.line(image,
                            tuple(np.multiply(joint_coords[0],[640,480]).astype(int)),
                            tuple(np.multiply(joint_coords[1],[640,480]).astype(int)),
                            (255, 0, 0), 5)
                    cv2.line(image, 
                            tuple(np.multiply(joint_coords[1], [640, 480]).astype(int)), 
                            tuple(np.multiply(joint_coords[2], [640, 480]).astype(int)), 
                            (255, 0, 0), 5)  # Line from middle joint to end joint
                    for joint in joint_coords:
                        cv2.circle(image, tuple(np.multiply(joint, [640, 480]).astype(int)), 8, (0, 255, 0), -1)

                    
                # Calculate angle for the selected exercise
                angle = calculate_angle(*joint_coords)
                last_angle = int(angle)

                # Rep count logic
                if exercise_name == "bicep_curl":
                    stage = "down"
                    if angle <= exercise["min_angle"] and stage == "down":
                        stage = "up"
                        rep_times.append(time.time())
                    if angle >= exercise["max_angle"] and stage == "up":
                        stage = "down"
                        counter += 1

                    if angle >= exercise["max_angle"]:
                        feedback = "Go up!"
                        color = (0, 0, 255)
                    elif angle <= exercise["min_angle"]:
                        feedback = "Go down!"
                        color = (0, 255, 0)
                    else:
                        feedback = "Keep going!"
                        color = (255, 255, 0)

                
                elif exercise_name == "squat":
                    stage = "up"
                    if angle <= exercise["min_angle"] and stage == "up":
                        stage = "down"
                        rep_times.append(time.time())
                    if angle >= exercise["max_angle"] and stage == 'down':
                        stage = "up"
                        counter +=1
                    if angle >= exercise["max_angle"]:
                        feedback = "Go down!"
                        color = (0, 0, 255)
                    elif angle <= exercise["max_angle"]:
                        feedback = "Go up!"
                        color = (0, 255, 0)
                    else:
                        feedback = "Keep going!"
                        color = (255, 255, 0)

                # Display rep count and workout time
                elapsed_time = int(time.time() - start_time)
                cv2.putText(
                    image,
                    f"Reps: {counter}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Angle: {int(angle)}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Time: {elapsed_time}s",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                
                cv2.putText(
                    image,
                    feedback,
                    (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # Fatigue detection
                if len(rep_times) > 1:
                    rep_differences = [rep_times[i] - rep_times[i - 1] for i in range (1, len(rep_times))]
                    avg_rep_time = sum(rep_differences) / len(rep_differences)
                    if avg_rep_time > fatigue_threshold:
                        cv2.putText(
                            image,
                            "Fatigue Detected!",
                            (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # Show image
            cv2.imshow("Exercise Tracker", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


