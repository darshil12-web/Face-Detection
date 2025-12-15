import streamlit as st
import face_recognition
import time
from PIL import Image
import numpy as np
import os
import zipfile
from io import BytesIO
import sqlite3
import uuid
from datetime import datetime
import json
from db import init_db


# Initialize database
conn = init_db()

# ---------------- CONFIG & INITIALIZATION ----------------
# Initialize all session state variables
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
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
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "shareable_link" not in st.session_state:
    st.session_state.shareable_link = None
if "session_loaded_from_url" not in st.session_state:
    st.session_state.session_loaded_from_url = False

st.set_page_config(page_title="Face Finder: Camera Detection", layout="wide")
st.title("Face Finder")
st.markdown("---")

# ---------------- HELPER FUNCTIONS ----------------

def save_photos_to_db(session_id, uploaded_files):
    """Save uploaded photos to database"""
    c = conn.cursor()
    
    # Create session entry
    c.execute(
        "INSERT OR IGNORE INTO sessions (session_id, created_at, threshold) VALUES (?, ?, ?)",
        (session_id, datetime.now(), st.session_state.MATCH_THRESHOLD)
    )
    
    # Save each photo
    for file in uploaded_files:
        file_bytes = file.getvalue()
        c.execute(
            "INSERT INTO photos (session_id, filename, file_data, uploaded_at) VALUES (?, ?, ?, ?)",
            (session_id, file.name, sqlite3.Binary(file_bytes), datetime.now())
        )
    
    conn.commit()
    return len(uploaded_files)

def get_photos_from_db(session_id):
    """Retrieve photos from database for a session"""
    c = conn.cursor()
    c.execute(
        "SELECT filename, file_data FROM photos WHERE session_id = ?",
        (session_id,)
    )
    photos = []
    for filename, file_data in c.fetchall():
        # Create a file-like object from bytes
        file_obj = BytesIO(file_data)
        file_obj.name = filename
        photos.append(file_obj)
    return photos

def save_match_to_db(session_id, target_face_data, target_name, matched_filenames):
    """Save match results to database"""
    c = conn.cursor()
    c.execute(
        "INSERT INTO matches (session_id, target_face_data, target_name, matched_photos, detected_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, sqlite3.Binary(target_face_data), target_name, json.dumps(matched_filenames), datetime.now())
    )
    conn.commit()

def find_matching_photos(target_encoding, comparison_files):
    """Find the target face in the photos"""
    matched_files = []
    if not comparison_files:
        return matched_files
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_threshold = st.session_state.MATCH_THRESHOLD
    
    for i, file in enumerate(comparison_files):
        status_text.text(f"Checking photo {i+1}/{len(comparison_files)}...")
        try:
            # Reset file pointer
            file.seek(0)
            image = face_recognition.load_image_file(file) 
            
            # Detect faces
            face_locations = face_recognition.face_locations(image, model="hog")
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            photo_matched = False
            if not face_encodings:
                continue
                
            for face_encoding in face_encodings:
                distance = face_recognition.face_distance([target_encoding], face_encoding)[0]
                if distance <= current_threshold:
                    photo_matched = True
                    break
            
            if photo_matched:
                matched_files.append({
                    "file": file,
                    "filename": file.name,
                    "faces_detected": len(face_encodings)
                })
        except Exception as e:
            st.warning(f"Error processing {file.name}: {str(e)}")
            pass
        
        progress_bar.progress((i + 1) / len(comparison_files))

    progress_bar.empty()
    status_text.empty()
    return matched_files

def create_zip_file(matched_photos):
    """Create a ZIP file of matched photos"""
    if not matched_photos:
        return None
        
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, match in enumerate(matched_photos):
            file_name = f"detected_{i+1}_{match['filename']}"
            match['file'].seek(0)
            zip_file.writestr(file_name, match['file'].read())
            
    zip_buffer.seek(0)
    return zip_buffer

def generate_shareable_link(session_id):
    """Generate a shareable link for the session"""
    # Get your server's base URL
    base_url = "http://localhost:8501"
    return f"{base_url}/?session={session_id}"

# ---------------- SIDEBAR: Database and Upload ----------------
with st.sidebar:
    st.header("Database Management")
    
    # Option to create new session or use existing
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Session", use_container_width=True, key="new_session_btn"):
            # Generate new session ID
            new_session_id = str(uuid.uuid4())[:8]
            st.session_state.current_session_id = new_session_id
            st.session_state.comparison_files = []
            st.session_state.shareable_link = generate_shareable_link(new_session_id)
            st.session_state.session_loaded_from_url = False
            st.rerun()
    
    with col2:
        session_input = st.text_input("Enter Session ID:", placeholder="Enter session ID", key="session_input")
        if session_input and st.button("Load Session", use_container_width=True, key="load_session_btn"):
            st.session_state.current_session_id = session_input
            st.session_state.session_loaded_from_url = False
            # Load photos from database
            photos = get_photos_from_db(session_input)
            if photos:
                st.session_state.comparison_files = photos
                st.success(f"Loaded {len(photos)} photos from session")
                st.session_state.shareable_link = generate_shareable_link(session_input)
                st.rerun()
            else:
                st.warning("No photos found for this session")
    
    # Display current session info
    if st.session_state.current_session_id:
        st.info(f"**Current Session:** {st.session_state.current_session_id}")
        # if st.session_state.shareable_link:
            # st.code(st.session_state.shareable_link, language="text")
    
    st.markdown("---")
    
    st.header("1. Upload All Photos")
    uploaded_files = st.file_uploader(
        "Upload Photos (JPG/PNG)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="all_photo_uploader"
    )
    
    if uploaded_files and st.session_state.current_session_id:
        if st.button("Save to Database", use_container_width=True, key="save_to_db_btn"):
            with st.spinner("Saving photos to database..."):
                saved_count = save_photos_to_db(st.session_state.current_session_id, uploaded_files)
            st.session_state.comparison_files = uploaded_files
            st.success(f"{saved_count} photos saved to database!")
            st.session_state.shareable_link = generate_shareable_link(st.session_state.current_session_id)
            st.rerun()
    
    # Display stored photos count
    if st.session_state.current_session_id:
        c = conn.cursor()
        c.execute(
            "SELECT COUNT(*) FROM photos WHERE session_id = ?",
            (st.session_state.current_session_id,)
        )
        count = c.fetchone()[0]
        st.caption(f"Stored in database: {count} photos")
    
    
    # Sensitivity setting
    # st.subheader("Detection Sensitivity")
    # st.session_state.MATCH_THRESHOLD = st.slider(
    #     "Matching Accuracy (Lower = Stricter)",
    #     min_value=0.30,
    #     max_value=0.70,
    #     value=0.50,
    #     step=0.05,
    #     help="Lower value = Stricter match (requires the face to look very similar)",
    #     key="threshold_slider"
    # )
    # Default threshold set (no UI)
    if "MATCH_THRESHOLD" not in st.session_state:
        st.session_state.MATCH_THRESHOLD = 0.50
    
    # Use anywhere in your code
    MATCH_THRESHOLD = st.session_state.MATCH_THRESHOLD


# ---------------- MAIN INTERFACE ----------------

# Show warning if no session is selected
if not st.session_state.current_session_id:
    st.warning("Please create or load a session from the sidebar to begin")
    
    # Quick start section
    st.markdown("---")
    st.header("Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create New Session", use_container_width=True, type="primary", key="quick_new_session"):
            new_session_id = str(uuid.uuid4())[:8]
            st.session_state.current_session_id = new_session_id
            st.session_state.shareable_link = generate_shareable_link(new_session_id)
            st.session_state.session_loaded_from_url = False
            st.rerun()
    
    with col2:
        st.write("")
    
else:
    # Display shareable link prominently
    if st.session_state.shareable_link:
        st.info(f"**Share this link with others:** `{st.session_state.shareable_link}`")
        st.markdown("---")
    
    if not st.session_state.comparison_files:
        st.warning("Please upload photos in the sidebar first to continue.")
    else:
        # --- STEP 2: Camera Capture ---
        st.header("2. Capture Target Face")
        st.info("Click below to activate camera and capture the face you want to find")
        
        if st.button("Activate Camera", use_container_width=True, type="primary", key="activate_camera_btn"):
            st.session_state.show_camera = True
        
        if st.session_state.show_camera:
            st.markdown("---")
            st.subheader("3. Capture the Face")
            
            # Camera input widget
            target_file = st.camera_input("Look at the camera and capture your face", key="camera_input")
            
            if target_file is not None:
                st.subheader("Captured Face:")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(target_file, width=300)
                
                with col2:
                    st.info("Processing captured face...")
                    
                    try:
                        target_file_bytes = target_file.getvalue()
                        image = face_recognition.load_image_file(BytesIO(target_file_bytes))
                        face_encodings = face_recognition.face_encodings(image)
                        
                        if len(face_encodings) > 0:
                            st.session_state.target_person_encoding = face_encodings[0]
                            st.session_state.target_person_name = "Target Person"
                            
                            st.success("Face successfully detected!")
                            
                            # --- STEP 4: Face Search ---
                            st.markdown("---")
                            st.header("4. Searching in Photos...")
                            
                            with st.spinner(f"Searching for the face in {len(st.session_state.comparison_files)} photos..."):
                                # Reset file pointers before processing
                                for file in st.session_state.comparison_files:
                                    file.seek(0)
                                matched_photos = find_matching_photos(
                                    st.session_state.target_person_encoding,
                                    st.session_state.comparison_files
                                )
                            
                            st.session_state.matched_photos = matched_photos
                            
                            # Save results to database
                            if matched_photos:
                                matched_filenames = [match['filename'] for match in matched_photos]
                                save_match_to_db(
                                    st.session_state.current_session_id,
                                    target_file_bytes,
                                    st.session_state.target_person_name,
                                    matched_filenames
                                )
                                st.success(f"Results saved to database!")
                            
                        else:
                            st.error("No face detected in the captured photo.")
                            st.session_state.target_person_encoding = None
                            
                    except Exception as e:
                        st.error(f"Error processing photo: {str(e)}")

# ---------------- 5. RESULTS SECTION ----------------
if st.session_state.matched_photos is not None:
    st.markdown("---")
    st.header("5. Results")
    matched_photos = st.session_state.matched_photos
    
    if matched_photos:
        st.success(f"Face found in **{len(matched_photos)} photos**")
        
        # Display results in tabs
        tab1, tab2 = st.tabs(["View Photos", "Statistics"])
        
        with tab1:
            # Display photos in a responsive grid
            cols = st.columns(3)
            for i, match in enumerate(matched_photos):
                with cols[i % 3]:
                    match['file'].seek(0)
                    st.image(match['file'], use_column_width=True)
                    st.caption(f"**{match['filename']}**")
                    st.caption(f"Faces detected: {match['faces_detected']}")
        
        with tab2:
            st.subheader("Detection Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Photos", len(st.session_state.comparison_files))
            
            with col2:
                st.metric("Matches Found", len(matched_photos))
            
            with col3:
                if len(st.session_state.comparison_files) > 0:
                    percentage = (len(matched_photos) / len(st.session_state.comparison_files)) * 100
                    st.metric("Match Rate", f"{percentage:.1f}%")
                else:
                    st.metric("Match Rate", "0%")
        
        # --- Download Option ---
        st.markdown("---")
        st.subheader("Download Results")
        
        zip_file = create_zip_file(matched_photos)
        if zip_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download All as ZIP",
                    data=zip_file,
                    file_name=f"detected_photos_{st.session_state.current_session_id}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="download_zip_btn"
                )
            
            with col2:
                # Option to save results to database
                if st.button("Save Results to Session", use_container_width=True, key="save_results_btn"):
                    st.success("Results already saved to database!")
    
    else:
        st.warning(f"No matches found in the photos")

# ---------------- SESSION HISTORY ----------------
with st.sidebar:
    st.markdown("---")
    # st.header("Session History")
    
    c = conn.cursor()
    # c.execute(
    #     "SELECT session_id, created_at FROM sessions ORDER BY created_at DESC LIMIT 5"
    # )
    recent_sessions = c.fetchall()
    
    if recent_sessions:
        for session_id, created_at in recent_sessions:
            try:
                # Try to parse the datetime
                if isinstance(created_at, str):
                    date_obj = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    date_obj = datetime.fromisoformat(str(created_at))
                date_str = date_obj.strftime("%b %d")
                
                # Get photo count
                c.execute("SELECT COUNT(*) FROM photos WHERE session_id = ?", (session_id,))
                result = c.fetchone()
                photo_count = result[0] if result else 0
                
                # Display session with quick load button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"{session_id[:6]}... • {photo_count} photos • {date_str}")
                with col2:
                    if st.button("↻", key=f"load_{session_id}", help=f"Load session {session_id}"):
                        st.session_state.current_session_id = session_id
                        st.session_state.session_loaded_from_url = False
                        photos = get_photos_from_db(session_id)
                        if photos:
                            st.session_state.comparison_files = photos
                            st.session_state.shareable_link = generate_shareable_link(session_id)
                            st.rerun()
            except Exception as e:
                st.caption(f"Session: {session_id[:6]}...")
    else:
        st.caption("No previous sessions")

# ---------------- RESET ----------------
st.markdown("---")
# col1, col2 = st.columns(2)

# with col1:
#     if st.button("Reset Current Session", use_container_width=True, key="reset_session_btn"):
#         st.session_state.comparison_files = []
#         st.session_state.matched_photos = None
#         st.session_state.target_person_encoding = None
#         st.session_state.show_camera = False
#         st.rerun()

# with col2:
#     if st.button("Clear All Data", use_container_width=True, type="secondary", key="clear_all_btn"):
#         # Clear database (optional - be careful with this)
#         c = conn.cursor()
#         c.execute("DELETE FROM photos")
#         c.execute("DELETE FROM sessions")
#         c.execute("DELETE FROM matches")
#         conn.commit()
        
#         # Clear session state
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
        
#         # Reset defaults
#         st.session_state.MATCH_THRESHOLD = 0.50
#         st.session_state.show_camera = False
#         st.rerun()

# ---------------- URL PARAMETER HANDLING ----------------
# Check for URL parameters (using st.experimental_get_query_params for older Streamlit versions)
try:
    # For Streamlit >= 1.28.0
    params = st.query_params
except AttributeError:
    # For older Streamlit versions
    params = st.experimental_get_query_params()

# Process URL parameters if any
if "session" in params and not st.session_state.session_loaded_from_url:
    session_id = params["session"][0] if isinstance(params["session"], list) else params["session"]
    
    # Set flag to prevent infinite reload
    st.session_state.session_loaded_from_url = True
    
    # Load the session
    st.session_state.current_session_id = session_id
    photos = get_photos_from_db(session_id)
    if photos:
        st.session_state.comparison_files = photos
        st.session_state.shareable_link = generate_shareable_link(session_id)
        st.success(f"Loaded session: {session_id} with {len(photos)} photos from URL")
        st.rerun()
    else:
        st.warning(f"No photos found for session: {session_id}")
