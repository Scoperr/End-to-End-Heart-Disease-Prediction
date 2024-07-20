from flask import Flask, render_template, request
import numpy as np
import pandas as np

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

#Route for Homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            physical_health_days = float(request.form.get('PhysicalHealthDays')),
            mental_health_days = float(request.form.get('MentalHealthDays')),
            sleep_hours = float(request.form.get('SleepHours')),
            height_in_meters = float(request.form.get('HeightInMeters')),
            weight_in_kilograms = float(request.form.get('WeightInKilograms')),
            bmi = float(request.form.get('BMI')),
            e_cigarette_usage = request.form.get('ECigaretteUsage'),
            race_ethnicity_category = request.form.get('RaceEthnicityCategory'),
            age_category = request.form.get('AgeCategory'),
            alcohol_drinkers = request.form.get('AlcoholDrinkers'),
            covid_pos = request.form.get('CovidPos'),
            had_angina = request.form.get('HadAngina'),
            had_stroke = request.form.get('HadStroke'),
            had_asthma = request.form.get('HadAsthma'),
            had_skin_cancer = request.form.get('HadSkinCancer'),
            had_copd = request.form.get('HadCOPD'),
            had_depressive_disorder = request.form.get('HadDepressiveDisorder'),
            had_kidney_disease = request.form.get('HadKidneyDisease'),
            had_arthritis = request.form.get('HadArthritis'),
            had_diabetes = request.form.get('HadDiabetes')
        )
        pred_df = data.get_data_as_dataframe()
        # print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
    
if __name__=="__main__":
    # while deploying remove debug as True
    app.run(host="0.0.0.0",debug=True)