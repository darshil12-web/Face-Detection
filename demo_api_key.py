import streamlit as st
from google import genai
from google.genai.errors import APIError
from PIL import Image
import io

# --- Initialization ---
try:
    # IMPORTANT: Ensure your GEMINI_API_KEY is set in Streamlit Secrets
    # (e.g., in a file named .streamlit/secrets.toml)
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, ValueError):
    st.error("Please set the 'GEMINI_API_KEY' in your Streamlit Secrets.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while initializing the Gemini client: {e}")
    st.stop()

# --- Gemini Analysis Function (No Change) ---

def analyze_image_with_gemini(target_person_image, comparison_image):
    """
    Uses Gemini to compare a target image with a comparison image
    to determine if the same person is present in both.
    """
    prompt = (
        "You have two images: a 'TARGET' image and a 'COMPARISON' image. "
        "Is the same person visible in the 'COMPARISON' image as the person in the 'TARGET' image? "
        "Respond with 'YES' if they are the same person, and 'NO' otherwise. "
        "Do not include any other text or explanation in your response."
    )
    
    try:
        target_img = Image.open(target_person_image)
        comparison_img = Image.open(comparison_image)

        contents = [
            prompt,
            target_img,
            comparison_img
        ]
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        
        return response.text.strip().upper() == "YES"
        
    except APIError as e:
        st.error(f"Gemini API Error during analysis: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during image processing: {e}")
        return False

st.title("Gemini Face Comparison and Filtering App")
st.markdown("---")

st.subheader("Step 1: Upload the Person to Look For (Target Photo)")
target_file = st.file_uploader(
    "Choose a clear photo of the target person:",
    type=["jpg", "jpeg", "png"],
    key="target_uploader"
)

if target_file:
    st.success("Target photo uploaded successfully!")
    st.image(target_file, caption="Target Person", width=200)

    st.subheader("Step 2: Upload a Set of Photos to Filter (Comparison Set)")
    comparison_files = st.file_uploader(
        "Choose up to 20 photos (select multiple files at once):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="comparison_uploader"
    )

    if comparison_files:
        st.info(f"Total {len(comparison_files)} photos uploaded for comparison.")
        
        if st.button("Find and Filter Target Person"):
            
            st.subheader("Results: Filtered Photos")
            
            with st.spinner(f"Using Gemini to analyze {len(comparison_files)} photos... This may take a moment."):
                found_count = 0
                
                cols = st.columns(3) 
                col_index = 0

                for i, comp_file in enumerate(comparison_files):
                    st.write(f"Analyzing photo {i+1} of {len(comparison_files)}...")
                    
                    is_match = analyze_image_with_gemini(target_file, comp_file)
                    
                    if is_match:
                        with cols[col_index % 3]:
                            st.image(comp_file, caption=f"Match #{found_count + 1}", width=250)
                            found_count += 1
                        
                        col_index += 1

            # Final Summary
            st.markdown("---")
            if found_count > 0:
                st.success(f"Success! The target person was found in **{found_count}** photos!")
            else:
                st.warning("Sorry, the target person was not found in any of the uploaded comparison photos.")
                
                
                
                
                
                
                
                