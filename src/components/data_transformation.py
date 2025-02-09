import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,FunctionTransformer,LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours','HeightInMeters', 'WeightInKilograms', 'BMI']
            categorical_columns = ['ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory','AlcoholDrinkers', 'CovidPos', 'HadAngina','HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD','HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis','HadDiabetes']
            
            age_mapping = {
                'Age 18 to 24': '18-34',
                'Age 25 to 29': '18-34',
                'Age 30 to 34': '18-34',
                'Age 35 to 39': '35-54',
                'Age 40 to 44': '35-54',
                'Age 45 to 49': '35-54',
                'Age 50 to 54': '35-54',
                'Age 55 to 59': '55-79',
                'Age 60 to 64': '55-79',
                'Age 65 to 69': '55-79',
                'Age 70 to 74': '55-79',
                'Age 75 to 79': '55-79',
                'Age 80 or older': '80+'
            }
            def map_age_category(X):
                # Assuming X is a DataFrame or a 2D array
                if isinstance(X, pd.DataFrame):
                    X = X.replace(age_mapping)
                else:
                    df = pd.DataFrame(X, columns=categorical_columns)
                    df = df.replace(age_mapping)
                    X = df.values
                return X
            age_category_transformer = FunctionTransformer(map_age_category)

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("age_mapper", age_category_transformer),
                    ("Ordinal_encoding", OrdinalEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical cols standard scaling completed.")
            logging.info("Categorical cols one hot encoding is completed")
            
            preprocesor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocesor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            selected_columns = [
                'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours',
                'HeightInMeters', 'WeightInKilograms', 'BMI',
                'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'AlcoholDrinkers', 'CovidPos',
                'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
                'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes'
            ]

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = train_df[selected_columns]
            test_df = test_df[selected_columns]

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "HadHeartAttack"

            X_train = train_df.drop(columns=[target_column_name],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name],axis=1)
            y_test = test_df[target_column_name]

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)