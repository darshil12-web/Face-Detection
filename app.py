import streamlit as st
import face_recognition
import time
from PIL import Image
import numpy as np
import os
import zipfile
from io import BytesIO

# ---------------- CONFIG & INITIALIZATION ----------------
if "MATCH_THRESHOLD" not in st.session_state:
    st.session_state.MATCH_THRESHOLD = 0.50
if "comparison_files" not in st.session_state:
    st.session_state.comparison_files = []
if "target_person_encoding" not in st.session_state:
    st.session_state.target_person_encoding = None
if "target_person_name" not in st.session_state:
    st.session_state.target_person_name = None
if "matched_photos" not in st.session_state:
    st.session_state.matched_photos = None
# New state for controlling the flow
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

st.set_page_config(page_title="Face Finder: Camera Detection", layout="wide")
st.title("Face Detect")
st.markdown("---")

# ---------------- FUNCTIONS ----------------

def find_matching_photos(target_encoding, comparison_files):
    """
    Finds the target face in the  photos.
    A match is returned only if the face is successfully detected (within the threshold).
    """
    matched_files = []
    if not comparison_files:
        return matched_files
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_threshold = st.session_state.MATCH_THRESHOLD
    
    for i, file in enumerate(comparison_files):
        status_text.text(f"Checking photo {i+1}/{len(comparison_files)}...")
        try:
            # Load the image file
            image = face_recognition.load_image_file(file) 
            
            # Detect faces
            face_locations = face_recognition.face_locations(image, model="hog")
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            photo_matched = False
            if not face_encodings:
                continue
                
            for face_encoding in face_encodings:
                distance = face_recognition.face_distance([target_encoding], face_encoding)[0]
                # Match found if distance is less than or equal to the threshold
                if distance <= current_threshold:
                    photo_matched = True
                    break # Stop checking other faces in this photo once a match is found
            
            if photo_matched:
                matched_files.append({
                    "file": file,
                    "faces_detected": len(face_encodings)
                })
        except Exception:
            # Skip corrupted or unreadable files
            pass
        
        progress_bar.progress((i + 1) / len(comparison_files))

    progress_bar.empty()
    status_text.empty()
    return matched_files 

def create_zip_file(matched_photos):
    """Creates a ZIP file of the matched photos."""
    if not matched_photos:
        return None
        
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, match in enumerate(matched_photos):
            # Get the file name, handling 'image.jpg' for camera input
            file_name_attr = getattr(match['file'], 'name', f"captured_image_{i+1}.jpg")
            file_name = f"detected_{i+1}_{file_name_attr}"
            zip_file.writestr(file_name, match['file'].getvalue())
            
    zip_buffer.seek(0)
    return zip_buffer

# Function to activate camera
def activate_camera():
    st.session_state.show_camera = True

# ---------------- SIDEBAR: Group Photo Upload ----------------
with st.sidebar:
    st.header("1.All Photos")
    uploaded_files = st.file_uploader(
        "Upload Photos (JPG/PNG)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="all_photo_uploader"
    )
    if uploaded_files:
        st.session_state.comparison_files = uploaded_files
        st.success(f"**{len(uploaded_files)}** photos loaded for comparison.")
    else:
        st.session_state.comparison_files = []
    
    # st.subheader("Detection Sensitivity")
    # st.session_state.MATCH_THRESHOLD = st.slider(
    #     "Matching Accuracy (Distance)",
    #     min_value=0.30,
    #     max_value=0.70,
    #     value=0.50, 
    #     step=0.05,
    #     help="Lower value = Stricter match (requires the face to look very similar)"
    # )
    # Default threshold set (no UI)
    if "MATCH_THRESHOLD" not in st.session_state:
        st.session_state.MATCH_THRESHOLD = 0.50
    
    # Use anywhere in your code
    MATCH_THRESHOLD = st.session_state.MATCH_THRESHOLD


# ---------------- MAIN INTERFACE: Target Capture Flow ----------------

if not st.session_state.comparison_files:
    st.warning("Please upload the photos in the sidebar first to continue.")
else:
    # --- STEP 2: Unique Link (Concept) ---
    st.header("2.Generate Link")
    st.info("To capture the target face, click the button below. This simulates accessing a unique link on a device with a camera.")

    # Using a button to simulate the action of clicking a unique link
    if st.button("Click Here to Capture Target Face", use_container_width=True) or st.session_state.show_camera:
        activate_camera() # Set state to show camera
        
        # --- STEP 3: Camera Input ---
        st.markdown("---")
        st.subheader("3.Capture the Face You Want to Find")
        
        # Camera input widget
        target_file = st.camera_input("Capture your face using the camera")

        # Processing Block
        if target_file is not None:
            st.subheader("Captured Face:")
            st.image(target_file, width=300)
            st.info("Creating face encoding...")
            
            try:
                image = face_recognition.load_image_file(target_file)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    st.session_state.target_person_encoding = face_encodings[0]
                    # Default name for camera input
                    st.session_state.target_person_name = getattr(target_file, 'name', "Captured Person")
                    st.success(f"Face of **{st.session_state.target_person_name}** detected.")

                    # --- STEP 4: Face Search ---
                    if st.session_state.comparison_files:
                        target_encoding = st.session_state.target_person_encoding
                        st.markdown("---")
                        st.header("4. Starting Search in Photos...")
                        with st.spinner(f"**Searching for the face in {len(st.session_state.comparison_files)} photos...**"):
                            matched_photos = find_matching_photos(
                                target_encoding,
                                st.session_state.comparison_files
                            )
                        st.session_state.matched_photos = matched_photos
                    
                else:
                    st.error("No face **detected** in the captured photo.")
                    st.session_state.target_person_encoding = None

            except Exception as e:
                st.error(f"Error processing photo: {str(e)}")

# ---------------- 5. Results and Download ----------------

if st.session_state.matched_photos is not None:
    st.markdown("---")
    st.header("5.Results: Photos Found")
    matched_photos = st.session_state.matched_photos
    
    if matched_photos:
        st.success(f"Face is present in **{len(matched_photos)} photos**.")
        st.subheader("Detected Photos:")
        
        # Display photos in a grid (3 columns)
        photo_cols = st.columns(3)
        for i, match in enumerate(matched_photos):
            with photo_cols[i % 3]:
                st.image(match["file"], use_column_width=True)
                st.markdown(f"**Detection Status:** **Present**")
                st.caption(f"Total faces in photo: {match['faces_detected']}")

        # --- Download Option ---
        st.markdown("---")
        st.subheader("Download Option")
        zip_file = create_zip_file(matched_photos)
        
        if zip_file:
            st.download_button(
                label="Download All Detected Photos as ZIP",
                data=zip_file,
                file_name=f"{st.session_state.target_person_name.replace(' ', '_')}_detected_photos.zip",
                mime="application/zip",
                use_container_width=True
            )
    else:
        st.warning(f"No photos were detected for **{st.session_state.target_person_name}**.")

# ---------------- RESET ----------------
st.markdown("---")
if st.button("Reset App and Start New Flow"):
    st.session_state.clear()
    st.session_state.MATCH_THRESHOLD = 0.50
    st.session_state.show_camera = False # Reset camera state
    st.rerun()