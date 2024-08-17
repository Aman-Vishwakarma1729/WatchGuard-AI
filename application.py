import os
from src.pipelines.prediction_pipeine import predict_on_video
from src.components.data_preprocessing import data_preprocessing
from tensorflow.keras.models import load_model
import streamlit as st
import time
import cv2


def main():
    st.markdown("""
                <style>
                [data-testid="sidebarTitle"] {
                color: white;
                }
                </style>
                """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Watch Demo Video", "About Project","Download Dataset"])

    if page == "Home":
        st.markdown("""
            <h1 style='text-align: center; color: white'>
                WATCH-GUARD-AI: Surveillance and Suspicious Activity Detection
            </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <style>
            .main {
                background-color: #454850;
            }
            h1 {
                color: #333;
                text-align: center;
                font-family: Arial, sans-serif;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 16px;
            }
            </style>
        """, unsafe_allow_html=True)
        
        input_video_dir_path = os.path.join(os.getcwd(), 'input_video')
        if not os.path.exists(input_video_dir_path):
            os.makedirs(input_video_dir_path)
        
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mpg"])
        
        if uploaded_file is not None:
            video_file_path = os.path.join(input_video_dir_path, uploaded_file.name)
            
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved file: {uploaded_file.name}")

            model = load_model(os.path.join(os.getcwd(), 'models', 'LCRN_Model.h5'))
            preprocess = data_preprocessing()
            image_height, image_width, sequence_length, _, classes_list = preprocess.set_dataset_variables()

            output_file_path = os.path.join(os.getcwd(), 'prediction_output', f"{uploaded_file.name.split('.')[0]}.mp4v")
            
            # Display spinner while prediction is ongoing
            with st.spinner("Wait... while prediction is made as it will take time based on video length"):
                predict_on_video(video_file_path, output_file_path, sequence_length, image_height, image_width, model, classes_list)
            
            time.sleep(2)

            if os.path.exists(output_file_path):
                st.markdown("---")
                st.write("The prediction will be displayed on top-left corner of the video.")
                st.video(output_file_path)
                st.write("Play the video to see the predictions.")
            else:
                st.error("Failed to play the video. Please try again.") 
                
            os.remove(video_file_path)
        
        # Add a section at the end with the GitHub link
        st.markdown("---")
        st.markdown("""
                    <h4 style='text-align: center; color: white'>
                    For more details about this project, visit the GitHub page.
                    </h4>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <h3><center><a href="https://github.com/Aman-Vishwakarma1729/WatchGuard-AI">GitHub</a></center></h3>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <center>&copy; 2024 Aman Vishwakarma</center>
                   """, unsafe_allow_html=True)

    elif page == "Watch Demo Video":
        st.markdown("<h1 style='text-align: center; color: white'>Demo Video", unsafe_allow_html=True)
        st.markdown("---")

        demo_video_path = os.path.join(os.getcwd(),'demo_video','059291870-fight-teenagers-street-yard-4k.mp4v')
        if os.path.exists(demo_video_path):
            st.video(demo_video_path)
        else:
            st.error("Demo video not found. Please ensure the video is in the correct location.")

        st.markdown("---")
        st.markdown("""
                    <h4 style='text-align: center; color: white'>
                    For more details about this project, visit the GitHub page.
                    </h4>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <h3><center><a href="https://github.com/Aman-Vishwakarma1729/WatchGuard-AI">GitHub</a></center></h3>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <center>&copy; 2024 Aman Vishwakarma</center>
                   """, unsafe_allow_html=True)

    elif page == "About Project":
        st.markdown("<h1 style='text-align: center; color: white'>About WATCH-GUARD-AI</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
            <p style='color: white'>
            WATCH-GUARD-AI is a surveillance system designed to detect suspicious activities in video footage. 
            Utilizing advanced deep learning techniques, the system can identify actions such as fighting, 
            running, and more, enabling real-time monitoring and alerting for enhanced security measures.
            </p>
            <p style='color: white'>
            The project leverages the LCRN model (Long-term Recurrent Convolutional Networks) for activity 
            recognition, combined with a user-friendly interface built with Streamlit. The system is capable 
            of processing uploaded videos, performing predictions, and displaying the results directly on the video.
            </p>
            <p style='color: white'>
            This project was developed as part of an effort to explore the potential of 
            AI in enhancing security systems. For more details and source code, visit the project's GitHub page.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
                    <h4 style='text-align: center; color: white'>
                    For more details about this project, visit the GitHub page.
                    </h4>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <h3><center><a href="https://github.com/Aman-Vishwakarma1729/WatchGuard-AI">GitHub</a></center></h3>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <center>&copy; 2024 Aman Vishwakarma</center>
                   """, unsafe_allow_html=True)
        
    elif page == "Download Dataset":
        st.markdown("<h1 style='text-align: center; color: white'>Download Dataset</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
            <p style='color: white'>
            You can download the dataset for this project from the following sources:
            </p>
            <ul>
                <li><a href="https://github.com/Aman-Vishwakarma1729/WatchGuard-AI/tree/main/data" target="_blank" style='color: #4CAF50'>Download from GitHub</a></li>
                <h3>OR</h3>
                <li>Download from Internet</li>
            </ul>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
                    <h4 style='text-align: center; color: white'>
                    For more details about this project, visit the GitHub page.
                    </h4>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <h3><center><a href="https://github.com/Aman-Vishwakarma1729/WatchGuard-AI">GitHub</a></center></h3>
                    """, unsafe_allow_html=True)
        st.markdown("""
                    <center>&copy; 2024 Aman Vishwakarma</center>
                   """, unsafe_allow_html=True)

if __name__ == "__main__":
  main()
