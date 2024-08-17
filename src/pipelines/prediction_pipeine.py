import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from IPython.display import display, Video

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


def predict_on_video(video_file_path, output_file_path, sequence_length, image_height, image_width, model, classes_list):
    # Initialize video reader and check if the video is opened successfully
    video_reader = cv2.VideoCapture(video_file_path)
    if not video_reader.isOpened():
        print(f"Error opening video: {video_file_path}")
        return

    # Get video properties
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    # Initialize video writer and check if it was created successfully
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (original_video_width, original_video_height))
    if not video_writer.isOpened():
        print(f"Error creating video writer for: {output_file_path}")
        video_reader.release()
        return

    frames_queue = []  
    predicted_class_name = ''
    

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (image_width, image_height))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        # Make predictions if the queue is filled
        if len(frames_queue) == sequence_length:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = classes_list[predicted_label]
            frames_queue.pop(0)  # Remove the first frame in the queue

        # Add prediction text to the frame
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write the frame with the prediction to the output video
        video_writer.write(frame)

    # Release the video reader and writer resources
    video_reader.release()
    video_writer.release()
