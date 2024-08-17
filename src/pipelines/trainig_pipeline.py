import os
import sys

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class model_training_config:
    mode_path = os.path.join(os.getcwd(),'models','LCRN_Model.h5')

class model_training:
    def __init__(self):
        self.model_training_config= model_training_config()

    def get_data_for_model_training(self,features,labels):
        try:
            one_hot_encoded_labels = to_categorical(labels)
            features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.25, shuffle = True, random_state = 5)
            logging.info(f"The training data size is {features_train.shape}")
            logging.info(f"The test data size is {features_test.shape}")
            features = None
            labels = None
        except Exception as e:
            logging.info("Exception occured in getting data for model training")
            raise CustomException(e,sys)
        return features_train, features_test, labels_train, labels_test
    
    
    def create_LRCN_model(self,SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST, features_train, labels_train):
        try:
            logging.info("Model training starts")

            model = Sequential()
            
            model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
            model.add(TimeDistributed(MaxPooling2D((4, 4))))
            
            model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
            model.add(TimeDistributed(MaxPooling2D((4, 4))))
            
            model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
            model.add(TimeDistributed(MaxPooling2D((2, 2))))
            
            model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
            model.add(TimeDistributed(MaxPooling2D((2, 2))))
                                            
            model.add(TimeDistributed(Flatten()))
                                            
            model.add(LSTM(32))
                                            
            model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

            logging.info(model.summary())

            early_stopping_callback = EarlyStopping(monitor = 'accuracy', patience = 10, mode = 'max', restore_best_weights = True)
            model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
            model_training_history = model.fit(x = features_train, y = labels_train, epochs = 70, batch_size = 4 , shuffle = True, validation_split = 0.25, callbacks = [early_stopping_callback])

            logging.info("Model training completed")

            model_path = self.model_training_config.mode_path
            model.save(model_path)

            logging.info(f"Model has been saved at:\n{model_path}\n")

        except Exception as e:
            logging.info("Exception occured while creating LRCN model")
            raise CustomException(e,sys)
        return model_path, model_training_history

