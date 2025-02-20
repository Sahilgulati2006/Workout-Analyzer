# AI-Powered Workout Tracker

## Overview

This project is an AI-powered workout tracker that uses your system's camera to analyze your workout in real time. It detects and counts repetitions of exercises, monitors form, and assesses fatigue by calculating the average time between repetitions.

## Features

- **Exercise Recognition:** Supports Bicep Curls, Squats, and Push-ups.
- **Rep Counting:** Automatically tracks completed repetitions.
- **Form Analysis:** Uses joint angles to provide real-time feedback.
- **Fatigue Detection:** Detects when a user is slowing down by analyzing rep timing.
- **Real-time Feedback:** Displays exercise data on the screen.

## Technologies Used

- **Python**: The core programming language.
- **OpenCV**: Used for capturing video and displaying feedback.
- **MediaPipe**: Detects and tracks human pose landmarks.
- **NumPy**: Used for mathematical calculations related to angle measurements.
- **Time Module**: Keeps track of elapsed workout time and rep timing.

## Installation

Ensure you have Python installed (preferably 3.7+). Then, install the required dependencies:

```sh
pip install opencv-python mediapipe numpy
```

## Usage

1. Run the script:

```sh
python Curl-Tracker.py
```

1. Select an exercise when prompted (`bicep_curl`, `squat`, or `push_up`).
2. Position yourself in front of the camera.
3. Follow the on-screen feedback for form corrections and fatigue detection.
4. Press `Q` to exit.

## How It Works

- The system captures video using OpenCV and detects key body landmarks with MediaPipe Pose.
- It calculates joint angles to determine movement phases.
- When the movement transitions through the predefined angles, a repetition is counted.
- The system records the time taken per rep and alerts the user if fatigue is detected (rep speed slows significantly).

## Future Improvements

- Add support for more exercises.
- Implement machine learning models for better movement analysis.
- Improve fatigue detection using moving averages.

## License

This project is open-source and available for modification and distribution.

## Author

Sahil Gulati, Brian Nguyen, Atharve Pandey
