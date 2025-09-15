import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import os
from datetime import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
BASE_DIR = '/content/drive/MyDrive/face_recognition' if os.path.exists('/content/drive') else 'face_recognition_data'
os.makedirs(BASE_DIR, exist_ok=True)
EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'embeddings.pth')
USER_IMAGE_DIR = os.path.join(BASE_DIR, 'saved_users')
os.makedirs(USER_IMAGE_DIR, exist_ok=True)
if 'known_embeddings' not in st.session_state:
    if os.path.exists(EMBEDDINGS_FILE):
        data = torch.load(EMBEDDINGS_FILE)
        st.session_state['known_embeddings'] = data['embeddings']
        st.session_state['known_names'] = data['names']
    else:
        st.session_state['known_embeddings'] = []
        st.session_state['known_names'] = []

def save_embeddings():
    torch.save({
        'embeddings': st.session_state['known_embeddings'],
        'names': st.session_state['known_names']
    }, EMBEDDINGS_FILE)

def get_embedding(face_img):
    with torch.no_grad():
        embedding = model(face_img.unsqueeze(0).to(device))
    return embedding.squeeze(0).cpu()

def recognize_faces(img_pil):
    draw = ImageDraw.Draw(img_pil)
    boxes, probs = mtcnn.detect(img_pil)

    if boxes is None or len(boxes) == 0:
        st.warning("No face detected in the image.")
        return img_pil

    faces = mtcnn.extract(img_pil, boxes, save_path=None)

    for box, face in zip(boxes, faces):
        embedding = get_embedding(face)
        identity = None
        min_dist = float('inf')

        for known_emb, name in zip(st.session_state['known_embeddings'], st.session_state['known_names']):
            dist = torch.dist(embedding, known_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 1.0:
            identity = "Unknown"

        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
        draw.text((x1, y1 - 10), identity, fill='lime')

    return img_pil

st.sidebar.header("Add New User")
name_input = st.sidebar.text_input("Name:")
image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

if st.sidebar.button("Add User"):
    if not name_input or not image_file:
        st.sidebar.error("Please provide both a name and an image.")
    else:
        img = Image.open(image_file).convert("RGB")
        boxes, probs = mtcnn.detect(img)

        if boxes is not None and len(boxes) == 1 and probs[0] > 0.95:
            face = mtcnn.extract(img, boxes, save_path=None)[0]
            embedding = get_embedding(face)

            st.session_state['known_embeddings'].append(embedding)
            st.session_state['known_names'].append(name_input)
            save_embeddings()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img.save(os.path.join(USER_IMAGE_DIR, f"{name_input}_{timestamp}.jpg"))

            st.sidebar.success(f"User '{name_input}' added successfully!")
        else:
            st.sidebar.error("Image must contain exactly one clear face.")

st.header("Face Recognition")
input_img = st.camera_input("Take a photo")

if input_img:
    img = Image.open(input_img).convert("RGB")
    result_img = recognize_faces(img)
    st.image(result_img, caption="Detected Faces")
