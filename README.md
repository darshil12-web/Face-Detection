# Face Finder: Camera Detection App

## 📸 Overview

The **Face Finder** is a Streamlit application designed to perform facial recognition in a batch of uploaded photos against a target face captured live using a webcam. It is optimized for speed using image resizing techniques and securely manages photo sessions and results using an SQLite database.

## ✨ Features

* **Session Management:** Create, load, and share sessions via URL parameters.
* **Database Integration:** Photos and match results are stored securely in a local SQLite database (`db.py` handles initialization).
* **Live Target Capture:** Use the webcam to capture the target face for comparison.
* **Optimized Face Search:** Uses the `face_recognition` library with image resizing (HOG model) for fast searching across large batches of photos.
* **Adjustable Sensitivity:** A slider allows users to adjust the matching threshold for stricter or more lenient detection.
* **Result Handling:** Displays matching photos, provides detection statistics, and offers a ZIP download of the results.

## ⚙️ Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Streamlit**
3.  **Required Libraries** (especially `face_recognition` which requires system libraries like `dlib` dependencies).

## 🚀 Setup and Installation

### 1. Clone the repository (If Applicable)

Assuming your code is in a project folder.

### 2. Install Python Dependencies

You will need the libraries mentioned in your script. The `face_recognition` library has specific dependencies.

```bash
# Install core libraries
pip install streamlit face-recognition pillow numpy sqlite3

# NOTE: Installing 'face-recognition' may require dlib prerequisites (like CMake and Visual C++ Build Tools on Windows, or build-essential on Linux).
