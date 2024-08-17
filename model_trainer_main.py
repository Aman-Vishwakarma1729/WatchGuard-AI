from src.components.data_preprocessing import data_preprocessing
from src.pipelines.trainig_pipeline import model_training
from src.pipelines.testing_pipeline import model_testing


data_preprocessing = data_preprocessing()
data_preprocessing.basic_data_info()
IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH,  DATASET_DIR, CLASSES_LIST = data_preprocessing.set_dataset_variables()
features, labels, video_files_paths = data_preprocessing.create_dataset(IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH,  DATASET_DIR, CLASSES_LIST)

model_trainer = model_training()
features_train, features_test, labels_train, labels_test = model_trainer.get_data_for_model_training(features, labels)
model_path, model_training_history = model_trainer.create_LRCN_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST, features_train, labels_train)

model_tester = model_testing()
model_tester.get_model_accuracy(features_test,labels_test)


