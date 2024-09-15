import streamlit as st
import requests

# Title of the interface
st.header("Weapon Detection Interface")
st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line separator

# Sidebar for the report
with st.sidebar:
    st.subheader("Detection Report")
    # Add a collapse button for the logs
    with st.expander("Show detection logs"):
        # Print "Log: Weapon detected" 16 times
        for i in range(16):
            st.markdown(f"Log {i+1}: Weapon detected")
    
    # Button to download the report
    # Horizontal line before the download button
    st.markdown("<hr>", unsafe_allow_html=True)  
    st.markdown("Download report")
    if st.button("Download"):
        st.markdown("Downloading report...")

# Create the layout in two columns with appropriate space
col1, col2 = st.columns([3, 1])

# Left column for the video
with col1:
    st.subheader("Camera View")
    # Placeholder for the video (currently blank), inside a div with class 'img-container'
    st.markdown("""
        <div class="img-container">
            <img src="http://127.0.0.1:5000/video_feed" alt="Video Stream">
        </div>
    """, unsafe_allow_html=True)