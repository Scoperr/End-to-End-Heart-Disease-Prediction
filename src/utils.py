import logging
import sys
import os

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,X_test,y_train,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            print(model)
            # gs = GridSearchCV(model, para, cv=3, scoring='accuracy')
            # gs.fit(X_train, y_train) 
            
            # model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Make Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
            logging.info(f"Model: {list(models.keys())[i]}")
            # logging.info(f"Best Parameters: {gs.best_params_}")
            logging.info(f"Train Accuracy: {train_model_score}")
            logging.info(f"Test Accuracy: {test_model_score}\n")
                
        return report
    except Exception as e:
        raise CustomException(e,sys)