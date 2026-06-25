# AirCanvas / SkyWrite

AirCanvas is a hand-tracking drawing application that uses a webcam, MediaPipe hand detection, and a pre-trained EMNIST model to run a gesture-driven canvas experience.

## Project Contents

- `Final_Code.py` — Main application source code.
- `my_emnist_model.h5` — Pre-trained EMNIST model weights required by the application.
- `font.ttf` — Font file used for on-screen rendering.
- `setup.bat` — Creates a Python virtual environment and installs the required dependencies.
- `Run_AirCanvas.bat` — Activates the virtual environment and starts the application.

## Requirements

- Windows OS
- Python 3.11+ (the project uses a virtual environment created by `setup.bat`)
- Webcam camera access

## Setup

1. Open a Command Prompt in this project folder.
2. Run `setup.bat`.
3. Wait until the setup completes.

This will create the `sky_env` virtual environment and install:

- `opencv-python`
- `mediapipe==0.10.11`
- `tensorflow==2.15.0`
- `Pillow`
- `protobuf==3.20.3`

## Running the App

After setup completes:

1. Run `Run_AirCanvas.bat`.
2. The application will activate the virtual environment and launch `Final_Code.py`.

## Notes

- `sky_env/` is a local virtual environment and should not be committed to GitHub.
- If you want to run the project manually without the batch files, use the `sky_env\Scripts\activate` script, then run:

```powershell
python Final_Code.py
```

## GitHub Publishing

This repository is ready for GitHub. If you already have a GitHub repository URL, add it as a remote and push:

```powershell
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

## Optional Improvements

- Add more documentation for app controls and gestures.
- Add a lighter configuration file for model paths.
- Publish the EMNIST model file externally if the GitHub repo must stay small.
