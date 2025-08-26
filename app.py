import streamlit as st
import os
import tempfile
import subprocess
import sys
import base64

# Set the page title and layout
st.set_page_config(page_title="Hazardous Video Classifier", layout="wide")

# Add custom CSS for layout
st.markdown("""
<style>
.small-video-container {
    max-width: 400px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

st.title("🚨 Hazardous Video Classification")
st.markdown("Upload a video to classify it as **Hazardous** or **Normal**.")

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'video_processed' not in st.session_state:
    st.session_state['video_processed'] = False
if 'inference_result' not in st.session_state:
    st.session_state['inference_result'] = None
if 'video_bytes' not in st.session_state:
    st.session_state['video_bytes'] = None

# Create a temporary directory
temp_dir = tempfile.mkdtemp()
os.makedirs(temp_dir, exist_ok=True)

# File uploader widget
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'mov', 'avi', 'mkv', 'webm'])

# Two columns: video preview + control panel
col1, col2 = st.columns([1, 2])  # Video column is narrower

with col1:
    if uploaded_file is not None:
        # Save uploaded file
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())

        st.session_state['video_path'] = video_path
        st.session_state['video_bytes'] = uploaded_file.getvalue()  # Store bytes for display

        st.markdown("**Uploaded Video Preview**")

        # Show video inside a responsive container
        video_base64 = base64.b64encode(st.session_state['video_bytes']).decode("utf-8")
        video_html = f"""
        <div style="display:flex; justify-content:center;">
          <video controls style="max-width:100%; width:350px; height:auto; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
          </video>
        </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)

        # Add a caption with file info
        st.caption(f"File: {uploaded_file.name} | Size: {len(st.session_state['video_bytes']) // 1024} KB")

with col2:
    st.markdown("**Control Panel**")

    # Run inference
    if st.session_state['video_path'] is not None and not st.session_state['video_processed']:
        if st.button('🔍 Predict', type="primary", use_container_width=True):
            with st.spinner('Analyzing video with AI model... This may take a moment.'):
                try:
                    result = subprocess.run(
                        [sys.executable, 'inference.py', st.session_state['video_path']],
                        capture_output=True, text=True, cwd=os.getcwd(),
                        timeout=300  # 5 minute timeout
                    )

                    # Store the result in session state for later use
                    st.session_state['inference_result'] = result

                    if result.returncode == 0:
                        # Parse inference.py output
                        output_lines = result.stdout.split('\n')
                        results = {}
                        for line in output_lines:
                            line = line.strip()
                            if 'Prediction:' in line:
                                results['prediction'] = line.split('Prediction:')[-1].strip()
                            elif 'Confidence:' in line:
                                conf_str = line.split('Confidence:')[-1].strip().replace('%', '')
                                try:
                                    results['confidence'] = float(conf_str) / 100
                                except ValueError:
                                    results['confidence'] = 0.0
                            elif 'Normal Score:' in line:
                                try:
                                    results['normal_score'] = float(line.split('Normal Score:')[-1].strip())
                                except ValueError:
                                    results['normal_score'] = 0.0
                            elif 'Hazard Score:' in line:
                                try:
                                    results['hazard_score'] = float(line.split('Hazard Score:')[-1].strip())
                                except ValueError:
                                    results['hazard_score'] = 0.0

                        st.session_state['prediction_result'] = results
                        st.session_state['video_processed'] = True
                    else:
                        st.error(f"Error running inference: {result.stderr}")

                except subprocess.TimeoutExpired:
                    st.error("❌ Analysis timed out. The video might be too long or the model is taking too long to process.")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.info("Please check that:")
                    st.info("1. inference.py is in the same directory")
                    st.info("2. The model file exists")
                    st.info("3. All dependencies are installed")

    # Display results
    if st.session_state['prediction_result'] is not None:
        results = st.session_state['prediction_result']

        st.success("✅ Analysis Complete!")
        st.markdown("### 📊 Classification Results")

        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric("Normal Score", f"{results.get('normal_score', 0):.4f}")
        with score_col2:
            st.metric("Hazard Score", f"{results.get('hazard_score', 0):.4f}")

        prediction = results.get('prediction', 'Unknown')
        confidence = results.get('confidence', 0)

        if 'Hazard' in prediction:
            st.error(f"**Prediction:** 🚨 {prediction} 🚨")
            st.error(f"**Confidence:** {confidence:.2%}")
        else:
            st.success(f"**Prediction:** ✅ {prediction}")
            st.success(f"**Confidence:** {confidence:.2%}")

        # Raw output
        with st.expander("📝 View Raw Output"):
            if st.session_state['inference_result'] is not None:
                st.text_area("Inference Output", 
                           st.session_state['inference_result'].stdout, 
                           height=200)
                if st.session_state['inference_result'].stderr:
                    st.text_area("Error Output",
                               st.session_state['inference_result'].stderr,
                               height=100)
            else:
                st.write("No raw output available")

# Reset button
if st.session_state['video_processed']:
    if st.button("🔄 Analyze Another Video", use_container_width=True):
        # Clear all session state
        for key in ['prediction_result', 'video_path', 'video_processed', 'inference_result', 'video_bytes']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by inference.py")
