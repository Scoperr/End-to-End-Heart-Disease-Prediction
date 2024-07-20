import logging
import sys
import os

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score,precision_score,recall_score
import mlflow
from mlflow.models import infer_signature
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

        
        MLFLOW_TRACKING_URI = "https://dagshub.com/ambaldhagetarun/Heart.mlflow"
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'ambaldhagetarun'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd9df774016ce00b3f7146677dd7233df6315b317'

        mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Heart_Attack_test")

        for name,model in models.items():
            
            with mlflow.start_run(run_name=name):
                print(name)
                model.fit(X_train, y_train)
            
                # Make Predictions
                y_test_pred = model.predict(X_test)
                
                test_model_score = accuracy_score(y_test, y_test_pred)
                
                report[name] = test_model_score

                
                mlflow.log_params(model.get_params())
                mlflow.log_metric("accuracy_score",test_model_score)
                
                signature = infer_signature(X_train, model.predict(X_train))

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train,
                    registered_model_name=name
                )

                logging.info(f"Model: {name}")
                logging.info(f"Test Accuracy: {test_model_score}\n")
            

            
                
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)