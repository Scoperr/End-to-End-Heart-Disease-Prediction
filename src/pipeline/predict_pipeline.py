import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __intit__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 physical_health_days, mental_health_days, sleep_hours,
                 height_in_meters, weight_in_kilograms, bmi, e_cigarette_usage,
                 race_ethnicity_category, age_category, alcohol_drinkers,
                 covid_pos, had_angina, had_stroke,
                 had_asthma, had_skin_cancer, had_copd, had_depressive_disorder,
                 had_kidney_disease, had_arthritis, had_diabetes):
        self.physical_health_days = physical_health_days
        self.mental_health_days = mental_health_days
        self.sleep_hours = sleep_hours
        self.height_in_meters = height_in_meters
        self.weight_in_kilograms = weight_in_kilograms
        self.bmi = bmi
        self.e_cigarette_usage = e_cigarette_usage
        self.race_ethnicity_category = race_ethnicity_category
        self.age_category = age_category
        self.alcohol_drinkers = alcohol_drinkers
        self.covid_pos = covid_pos
        self.had_angina = had_angina
        self.had_stroke = had_stroke
        self.had_asthma = had_asthma
        self.had_skin_cancer = had_skin_cancer
        self.had_copd = had_copd
        self.had_depressive_disorder = had_depressive_disorder
        self.had_kidney_disease = had_kidney_disease
        self.had_arthritis = had_arthritis
        self.had_diabetes = had_diabetes
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "PhysicalHealthDays": [self.physical_health_days],
                "MentalHealthDays": [self.mental_health_days],
                "SleepHours": [self.sleep_hours],
                "HeightInMeters": [self.height_in_meters],
                "WeightInKilograms": [self.weight_in_kilograms],
                "BMI":  [self.bmi],
                "ECigaretteUsage": [self.e_cigarette_usage],
                "RaceEthnicityCategory": [self.race_ethnicity_category],
                "AgeCategory": [self.age_category],
                "AlcoholDrinkers": [self.alcohol_drinkers],
                "CovidPos": [self.covid_pos],
                "HadAngina": [self.had_angina],
                "HadStroke": [self.had_stroke],
                "HadAsthma": [self.had_asthma],
                "HadSkinCancer": [self.had_skin_cancer],
                "HadCOPD": [self.had_copd],
                "HadDepressiveDisorder": [self.had_depressive_disorder],
                "HadKidneyDisease": [self.had_kidney_disease],
                "HadArthritis": [self.had_arthritis],
                "HadDiabetes": [self.had_diabetes]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)