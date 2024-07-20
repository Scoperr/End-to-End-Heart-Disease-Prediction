from random import randint, uniform
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_models
import os
import sys



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, X_train,X_test,y_train,y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier()
            }


            # params = {
            #     "Decision Tree": {
            #         'criterion': ['gini', 'entropy'],
            #         'splitter': ['best', 'random'],
            #         'max_features': ['sqrt', 'log2', None]
            #     },
            #     "Random Forest": {
            #         'n_estimators': [10, 20, 30, 40, 50],
            #         'criterion': ['gini', 'entropy'],
            #         'max_features': ['sqrt', 'log2', None]
            #     },
            #     "Gradient Boosting": {
            #         'learning_rate': [0.1, 0.05, 0.01],
            #         'subsample': [0.8, 0.9, 1.0],
            #         'n_estimators': [10, 20, 30, 40, 50]
            #     },
            #     "Logistic Regression": {
            #         'penalty': ['l1', 'l2'],
            #         'C': [0.1, 1, 10],
            #         'solver': ['liblinear', 'saga']
            #     },
            #     "AdaBoostClassifier": {
            #         'learning_rate': [0.1, 0.5],
            #         'n_estimators': [10, 20, 30, 40, 50]
            #     },
            #     "KNeighborsClassifier": {
            #         'n_neighbors': [3, 5, 7],
            #         'weights': ['uniform', 'distance'],
            #         'algorithm': ['auto', 'ball_tree']
            #     },
            #     "Support Vector Classifier": {
            #         'C': [0.1, 1, 10],
            #         'kernel': ['linear', 'rbf'],
            #         'gamma': ['scale', 'auto']
            #     }
            # }

            model_report: dict = evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models
            )

            # Find the best model based on test accuracy
            print(model_report)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            # Check if the best model meets the performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            # Get the best model from the models dictionary
            best_model = models[best_model_name]


            logging.info("Best model found on both training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and evaluate the best model
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy
        except Exception as e:
            raise CustomException(e,sys)