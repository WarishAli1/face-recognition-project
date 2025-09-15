# Face Recognition App

This project is a **real-time face recognition system** built using Python, Streamlit, and PyTorch. It allows users to add new faces and recognize them using a webcam interface.

## Motivation

I recently learned **Convolutional Neural Networks (CNNs)** and wanted to apply my knowledge in a **real-world application**. Due to hardware limitations, I used a **pre-trained model** (`InceptionResnetV1` from the `facenet-pytorch` library) for feature extraction instead of training a model from scratch. This approach allows the app to run efficiently without a powerful GPU.

## Features

- Add new users with their name and image.
- Save embeddings of known faces for future recognition.
- Real-time face recognition via webcam input.
- Display recognized faces with bounding boxes and names.

## How It Works

1. The app uses **MTCNN** to detect faces in images.
2. Detected faces are passed through a **pre-trained InceptionResnetV1 model** to extract embeddings (feature vectors).
3. These embeddings are compared with **stored embeddings** to identify the person.
4. If a new face is added, its embedding and name are saved for future recognition.
5. Faces are displayed with bounding boxes and the recognized name.

## Technical Details

- **Backend:** PyTorch, `facenet-pytorch`
- **Frontend:** Streamlit
- **Data Storage:** User images and embeddings are stored in **Google Drive** to persist across sessions.
- **Pre-trained Model:** `InceptionResnetV1` trained on the VGGFace2 dataset.
- **Recognition Threshold:** Distance > 1.0 is considered "Unknown".

## Setup Instructions

1. Install required libraries:

```bash
pip install torch torchvision
pip install facenet-pytorch streamlit
pip install opencv-python-headless
```

2. Mount Google Drive to store embeddings and user images:

```bash
from google.colab import drive
drive.mount('/content/drive')
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Use the webcam interface to add users and recognize faces.

Notes

- This project does not train a CNN from scratch; it uses pre-trained embeddings due to hardware limitations.

- User images and embeddings are stored permanently in Google Drive, ensuring data is available across Colab sessions.

- This app demonstrates how CNNs can be applied in real-world computer vision applications.

