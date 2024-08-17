import os
import sys
import cv2
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class data_preprocessing_config:
    dataset_path = os.path.join(os.getcwd(),'data')
    extracted_dataset_path = os.path.join(os.getcwd(),'artifacts','extracted_data.csv')

class data_preprocessing:
    def __init__(self):
        self.data_preprocessing_config = data_preprocessing_config()
    
    def basic_data_info(self):
        try:
            dataset_path = self.data_preprocessing_config.dataset_path
            logging.info(f"Got the acess to data at:\n{dataset_path}\n")
            logging.info(f"Information about data:\n")
            for folder_name in os.listdir(dataset_path):
                folder_path = os.path.join(dataset_path, folder_name)
                if not os.path.isdir(folder_path):
                   continue
                video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi", ".mov",".mpg"))]
                logging.info(f"For the {folder_name} data we have {len(video_files)} videos")
                if not video_files:
                    print(f"No video files found for: {folder_name}")
                    continue
        except Exception as e:
            logging.info("Exception occured in getting basic dataset information")
            raise CustomException(e,sys)
            
    def set_dataset_variables(self):
        try:
            IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
            SEQUENCE_LENGTH = 30
            DATASET_DIR = self.data_preprocessing_config.dataset_path
            CLASSES_LIST = []
            for folder_name in os.listdir(DATASET_DIR):
                CLASSES_LIST.append(folder_name)
            logging.info("Set the dataset variabels")
        except Exception as e:
            logging.info("Exception occured in setting dataset variables")
            raise CustomException(e,sys)
        return IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH,  DATASET_DIR, CLASSES_LIST
    
    def frames_extraction(self,video_path,IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH):
        try:
            frames_list = []
            video_reader = cv2.VideoCapture(video_path)
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
            for frame_counter in range(SEQUENCE_LENGTH):
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window) 
                success, frame = video_reader.read() 
                if not success:
                    break
                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                normalized_frame = resized_frame / 255
                frames_list.append(normalized_frame)
            video_reader.release()
        except Exception as e:
            logging.info("Exception occured in frame extraction")
            raise CustomException(e,sys)
        return frames_list
    
    def create_dataset(self,IMAGE_HEIGHT, IMAGE_WIDTH,SEQUENCE_LENGTH,DATASET_DIR,CLASSES_LIST):
        try:
            logging.info("Process of data-creation started")
            features = []
            labels = []
            video_files_paths = []
            for class_index, class_name in enumerate(CLASSES_LIST):
                logging.info(f'Extracting Data of Class: {class_name}')
                files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
                for file_name in files_list:
                    video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
                    frames = self.frames_extraction(video_file_path,IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH)
                    if len(frames) == SEQUENCE_LENGTH:
                        features.append(frames)
                        labels.append(class_index)
                        video_files_paths.append(video_file_path)
            features = np.asarray(features)
            labels = np.array(labels)
        except Exception as e:
            logging.info("Exception occured while creating dataset")
            raise CustomException(e,sys)
        return features, labels, video_files_paths
    
        
        
                
        