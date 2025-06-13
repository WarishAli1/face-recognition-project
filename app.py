import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=14, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

EMBEDDINGS_FILE = 'embeddings.pth'

# Load or initialize
if 'known_embeddings' not in st.session_state:
    if os.path.exists(EMBEDDINGS_FILE):
        data = torch.load(EMBEDDINGS_FILE)
        st.session_state['known_embeddings'] = data['embeddings']
        st.session_state['known_names'] = data['names']
    else:
        st.session_state['known_embeddings'] = []
        st.session_state['known_names'] = []

# Save embeddings
def save_embeddings():
    torch.save({
        'embeddings': st.session_state['known_embeddings'],
        'names': st.session_state['known_names']
    }, EMBEDDINGS_FILE)

# Extract face embedding
def get_embedding(img):
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) != 1 or probs[0] < 0.95:
        return None  # No face, multiple faces, or low confidence

    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None

    img_embedding = model(img_cropped.unsqueeze(0).to(device))
    return img_embedding.squeeze(0).detach().cpu()


# Compare with known embeddings
def recognize_face(img):
    embedding = get_embedding(img)
    if embedding is None:
        return "No face detected"

    min_dist = float('inf')
    identity = "Unknown"
    for known_emb, name in zip(st.session_state['known_embeddings'], st.session_state['known_names']):
        dist = torch.dist(embedding, known_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name if dist < 1.0 else "Unknown"

    return f"{identity} (dist: {min_dist:.3f})"

# Add new user
st.sidebar.header("Add New User")
name_input = st.sidebar.text_input("Name:")
image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

if st.sidebar.button("Add User"):
    if name_input and image_file:
        img = Image.open(image_file).convert("RGB")
        embedding = get_embedding(img)
        if embedding is not None:
            st.session_state['known_embeddings'].append(embedding)
            st.session_state['known_names'].append(name_input)
            save_embeddings()
            st.sidebar.success(f"User '{name_input}' added!")
        else:
            st.sidebar.error("No face detected in image.")
    else:
        st.sidebar.error("Please provide a name and an image.")

# Face Recognition
st.header("Face Recognition")
input_img = st.camera_input("Take a photo")

if input_img:
    img = Image.open(input_img).convert("RGB")
    result = recognize_face(img)
    st.image(img, caption=result)
