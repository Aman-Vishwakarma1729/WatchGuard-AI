import os
from src.pipelines.prediction_pipeine import predict_on_video
from src.components.data_preprocessing import data_preprocessing
from tensorflow.keras.models import load_model


preprocess = data_preprocessing()
image_height, image_width,  sequence_length,  DATASET_DIR, classes_list = preprocess.set_dataset_variables()

model = load_model(os.path.join(os.getcwd(),'models','LCRN_Model.h5'))

input_video_dir_path = os.path.join(os.getcwd(),'input_video')

if not os.path.exists(input_video_dir_path):
    os.makedirs(input_video_dir_path)
    print(f"Created directory: {input_video_dir_path}")
else:
    print(f"Directory already exists: {input_video_dir_path}")

video_files_name = [f for f in os.listdir(input_video_dir_path) if f.lower().endswith((".mp4", ".avi", ".mov",".mpg"))]

if len(video_files_name) > 1:
    print('The input_video folder has more than 1 input video which is leading to confusion please make sure it has only 1 video on which you want to test the model')
elif len(video_files_name) == 0:
    print("The input_video folder is empty")
else:
    video_file_name = video_files_name[0]

video_file_path = os.path.join(input_video_dir_path,video_file_name)

output_file_path = os.path.join(os.getcwd(),'prediction_output',video_file_name)


predict_on_video(video_file_path, output_file_path, sequence_length, image_height, image_width, model, classes_list)

for video_file in video_files_name:
    file_path = os.path.join(input_video_dir_path, video_file)
    try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    except Exception as e:
        print(f"Error removing file: {file_path}. Error: {str(e)}")