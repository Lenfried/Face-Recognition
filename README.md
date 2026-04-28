# Face-Recognition
Face Recognition With Raspberry Pi and OpenCV for use in the CISE Summer Program

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-OS-C51A4A?logo=raspberrypi&logoColor=white)

## Overview
This project uses OpenCV and face_recognition to capture images, train a model, and perform real-time face recognition.

## Setup

### Create Virtual Environment
python3 -m venv --system-site-packages face_rec

### Activate Environment
source face_rec/bin/activate

### Update System
sudo apt update
sudo apt full-upgrade

### Install Dependencies
pip install -r requirements.txt

## Project Structure
- dataset/
- image_capture.py
- train_model.py
- recognize.py
- requirements.txt

## Usage

### Capture Images
Run image_capture.py to collect training data.

### Train Model
Run train_model.py.

### Run Recognition
Run recognize.py.
