import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import time

# ---------------- CONFIG ----------------
MATCH_THRESHOLD = 0.45   # Lower = stricter (0.4–0.6 best)
# ---------------------------------------

st.set_page_config(page_title="Real Face Detection", page_icon="", layout="wide")

st.title("Face Detection App")

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------- FUNCTIONS ----------------

def load_face_encoding(image_file):
    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return None
    return encodings[0]

def compare_faces(target_encoding, comparison_files):
    matches = []

    progress = st.progress(0)
    status = st.empty()

    for i, file in enumerate(comparison_files):
        status.text(f"Processing {i+1}/{len(comparison_files)} : {file.name}")

        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            continue

        distance = face_recognition.face_distance([target_encoding], encodings[0])[0]
        confidence = round((1 - distance) * 100, 2)

        if distance <= MATCH_THRESHOLD:
            matches.append({
                "file": file,
                "distance": round(distance, 4),
                "confidence": confidence
            })

        progress.progress((i + 1) / len(comparison_files))
        time.sleep(0.05)

    progress.empty()
    status.empty()
    return matches

# ---------------- UI ----------------

col1, col2 = st.columns(2)

with col1:
    target_file = st.file_uploader("Upload Target Image", type=["jpg", "png", "jpeg"])

    if target_file:
        st.image(target_file, use_column_width=True)

with col2:
    st.subheader("Images to Compare")
    comparison_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

st.markdown("---")

if st.button("Face Matching", use_container_width=True):
    if not target_file or not comparison_files:
        st.error("Please upload both the target image and the comparison images.")
    else:
        with st.spinner("Face Detect Running..."):
            target_encoding = load_face_encoding(target_file)

            if target_encoding is None:
                st.error("No face was detected in the target image.")
            else:
                results = compare_faces(target_encoding, comparison_files)
                st.session_state.results = results

# ---------------- RESULTS ----------------

if st.session_state.results:
    st.markdown("---")
    st.success(f"{len(st.session_state.results)} PERFECT MATCH(es) found!")

    cols = st.columns(3)
    for i, match in enumerate(st.session_state.results):
        with cols[i % 3]:
            st.image(match["file"], caption="Matched Face", use_column_width=True)
            st.write(f"**Confidence:** {match['confidence']}%")
            st.write(f"**Distance:** {match['distance']}")

elif target_file and comparison_files:
    st.info("No perfect match has been found so far.")

# ---------------- RESET ----------------
st.markdown("---")
if st.button("Reset"):
    st.session_state.clear()
    st.rerun()
