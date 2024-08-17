import os
from src.pipelines.prediction_pipeine import predict_on_video
from src.components.data_preprocessing import data_preprocessing
from tensorflow.keras.models import load_model
import streamlit as st
import time


def main():
    st.title("WATCH-GUARD-AI: Surveillance and Suspicious Activity Detection")
    
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

        output_file_path = os.path.join(os.getcwd(), 'prediction_output', f"{uploaded_file.name.split(".")[0]}.mp4v")
        
        predict_on_video(video_file_path, output_file_path, sequence_length, image_height, image_width, model, classes_list)
        
        time.sleep(2)

        if os.path.exists(output_file_path):
            st.write("The prediction will be displayed on top-left corner of video")
            st.video(output_file_path)
            st.write("Play the video to see the predictions")
        else:
            st.error("Failed to play the video. Please try again.")
            
        os.remove(video_file_path)

if __name__ == "__main__":
    main()
