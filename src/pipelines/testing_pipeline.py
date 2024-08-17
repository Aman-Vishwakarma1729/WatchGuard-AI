import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class model_testing_config:
    mode_path = os.path.join(os.getcwd(),'models','LCRN_Model_try.h5')

class model_testing:
    def __init__(self):
        self.model_testing_config= model_testing_config()

    def get_model_accuracy(self,features_test,labels_test):
        try:
            model = load_model(self.model_testing_config.mode_path)
            acc = 0
            for i in range(len(features_test)):
                predicted_label = np.argmax(model.predict(np.expand_dims(features_test[i],axis =0))[0])
                actual_label = np.argmax(labels_test[i])
                if predicted_label == actual_label:
                    acc += 1
            acc = (acc * 100)/len(labels_test)
            logging.info(f"The model accuracy is: {acc}%",)
        except Exception as e:
            logging.info("Exception occured while calculating model accuracy")
            raise CustomException(e,sys)