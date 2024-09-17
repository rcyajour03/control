<<<<<<< HEAD
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import general_diseases as gd
from general_diseases import Diagnose
import special_diseases as sd
from special_diseases import Alzheimer
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import pandas as pd
import pickle
from interpret import show
import re
import io
import sys
from matplotlib import pyplot as plt
from io import BytesIO


# Capture the output of show() to extract the URL
class StreamToLogger(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = None

    def write(self, message):
        super().write(message)
        # Check if the message contains the URL pattern
        match = re.search(r'http://\S+', message)
        if match:
            self.url = match.group(0)


# Load pre-trained models
diabetes_model = pickle.load(open("Diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_disease_model.sav", "rb"))
alzheimer_model = pickle.load(open("alzheimer_model.sav", "rb"))

# Sidebar menu for user navigation
with st.sidebar:
    option = option_menu("Health Assistant", 
                         ["General", "Alzheimer", "Diabetes", "Heart Disease"], 
                         default_index=0)

# General Disease Section
if option == "General":
    st.title("General Disease Prediction")
    available_diseases = gd.find_diseases.diseases()

    chosen_disease = st.selectbox("Select a disease to check:", available_diseases)
    st.write(f"Diagnosing {chosen_disease}:")
    
    symptoms = Diagnose.diagnose(chosen_disease)
    st.write("For each symptom, enter 1 if you have it, otherwise enter 0.")
    
    user_input = []
    for symptom in symptoms:
        user_input.append(st.text_input(symptom))
    
    if st.button("Submit"):
        input_data = [int(i) for i in user_input]
        result = Diagnose.Prediction(chosen_disease, symptoms, input_data)
        
        if result == 1:
            st.warning(f"Alert: You may be prone to {chosen_disease}.")
        else:
            st.success(f"Good news! You are not at risk for {chosen_disease}.")

# Alzheimer Section
elif option == "Alzheimer":
    st.title("Alzheimer Disease Prediction")

    info = pd.DataFrame({
        "Feature": ["Sex", "Age", "Education", "SES", "CDR", "MMSE", "ETIV", "NWBV", "ASF"],
        "Description": ["Gender", "Age in years", "Years of education", "Socioeconomic Status", 
                        "Clinical Dementia Rating", "Mini Mental State Exam", 
                        "Estimated Total Intracranial Volume", "Normalized Whole Brain Volume", 
                        "Atlas Scaling Factor"]
    })
    st.table(info)

    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.text_input('Sex')
        ses = st.text_input('Socioeconomic Status')
        etiv = st.text_input('Estimated Total Intracranial Volume')

    with col2:
        age = st.text_input('Age')
        cdr = st.text_input('Clinical Dementia Rating')
        nwbv = st.text_input('Normalized Whole Brain Volume')

    with col3:
        educ = st.text_input('Years of Education')
        mmse = st.text_input('Mini Mental State Exam')
        asf = st.text_input('Atlas Scaling Factor')

    if st.button("Alzheimer Test Result"):
        input_data = [sex, age, educ, ses, mmse, cdr, etiv, nwbv, asf]
        
        # input_data = ['52', '1', '3', '145', '233', '1', '0', '150', '0']
        # input_data = [i for i in input_data]
        #                  OR
        input_data = [eval(i) for i in input_data]
        
        # Make prediction
        prediction = alzheimer_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')
        
        # Redirect stdout to capture output from show
        logger = StreamToLogger()
        sys.stdout = logger


        explainer = alzheimer_model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
        show(explainer) # This opens the explanation in a browser tab

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the extracted URL
        iframe_url = logger.url

        # Embed the generated URL in an iframe if available
        if iframe_url:
            st.components.v1.iframe(src=iframe_url, width=600, height=800)
        else:
            st.write("No URL found for the decision tree explanation.")


    if st.button("Show statistics"):
        avg_stats = sd.Alzheimer.alzheimer()
        user_data = [int(sex), int(age), int(educ), int(ses), int(mmse), int(cdr), int(etiv), int(nwbv), int(asf)]
        features = ["Age", "Education", "SES", "MMSE", "CDR", "ETIV", "NWBV", "ASF"]

        for i, feature in enumerate(features):
            plt.figure()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Average", "User"], [avg_stats[i], user_data[i]], color=["#902fed", "yellow"])
            ax.set_title(feature)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

# Diabetes Section
elif option == "Diabetes":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.text_input('Number of Pregnancies')
        skin_thickness = st.text_input('Skin Thickness')

    with col2:
        glucose = st.text_input('Glucose Level')
        insulin = st.text_input('Insulin Level')

    with col3:
        blood_pressure = st.text_input('Blood Pressure')
        bmi = st.text_input('BMI')

    with col1:
        dpf = st.text_input('Diabetes Pedigree Function')
        age = st.text_input('Age')

    if st.button("Diabetes Test Result"):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        
        input_data = [eval(i) for i in input_data]
        
        # Make prediction
        prediction = diabetes_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')
        
        # Redirect stdout to capture output from show
        logger = StreamToLogger()
        sys.stdout = logger


        explainer = diabetes_model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
        show(explainer) # This opens the explanation in a browser tab

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the extracted URL
        iframe_url = logger.url

        # Embed the generated URL in an iframe if available
        if iframe_url:
            st.components.v1.iframe(src=iframe_url, width=600, height=800)
        else:
            st.write("No URL found for the decision tree explanation.")


    if st.button("Show statistics"):
        avg_stats = sd.Diabetes.diabetes()
        user_data = [int(pregnancies), int(glucose), int(blood_pressure), int(skin_thickness), int(insulin), int(bmi), int(dpf), int(age)]
        features = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]

        for i, feature in enumerate(features):
            plt.figure()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Average", "User"], [avg_stats[i], user_data[i]], color=["#902fed", "yellow"])
            ax.set_title(feature)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

# Heart Disease Section
elif option == "Heart Disease":
    st.title("Heart Disease Prediction")

    info = pd.DataFrame({
        "Feature": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max HR", 
                    "Exercise Induced Angina", "ST Depression", "Peak Exercise Slope", "Major Vessels", "Thalassemia"],
        "Description": ["Age in years", "Sex", "Type of chest pain", "Resting blood pressure", "Cholesterol in mg/dl", 
                        "Fasting blood sugar > 120 mg/dl", "Resting electrocardiographic results", 
                        "Maximum heart rate achieved", "Exercise-induced angina", "ST depression", 
                        "Slope of peak exercise ST segment", "Number of major vessels colored by fluoroscopy", 
                        "Thalassemia: 0=Normal, 1=Fixed defect, 2=Reversible defect"]
    })
    st.table(info)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting BP')

    with col2:
        sex = st.text_input('Sex')
        chol = st.text_input('Cholesterol')

    with col3:
        cp = st.text_input('Chest Pain Type')
        fbs = st.text_input('Fasting Blood Sugar')

    with col1:
        restecg = st.text_input('Resting ECG')
        thalach = st.text_input('Max HR')

    with col2:
        exang = st.text_input('Exercise Induced Angina')
        oldpeak = st.text_input('ST Depression')

    with col3:
        slope = st.text_input('Slope of Peak Exercise ST Segment')
        ca = st.text_input('Number of Major Vessels')
        thal = st.text_input('Thalassemia')

    
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Prediction and result display
    if st.button('Heart Disease Test Result'):
        # input_data = ['52', '1', '3', '145', '233', '1', '0', '150', '0', '2.3', '1','0', '2']
        # input_data = [i for i in input_data]
        #                  OR
        input_data = [eval(i) for i in input_data]
        
        # Make prediction
        prediction = heart_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')
        
        # Redirect stdout to capture output from show
        logger = StreamToLogger()
        sys.stdout = logger


        explainer = heart_model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
        show(explainer) # This opens the explanation in a browser tab

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the extracted URL
        iframe_url = logger.url

        # Embed the generated URL in an iframe if available
        if iframe_url:
            st.components.v1.iframe(src=iframe_url, width=600, height=800)
        else:
            st.write("No URL found for the decision tree explanation.")

    if st.button("Show statistics"):
        avg_stats = sd.Heart.heart()
        user_data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), int(oldpeak), int(slope), int(ca), int(thal)]
        features = ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max HR", "Exercise Angina", "ST Depression", "Slope", "Major Vessels", "Thalassemia"]

        for i, feature in enumerate(features):
            plt.figure()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Average", "User"], [avg_stats[i], user_data[i]], color=["#902fed", "yellow"])
            ax.set_title(feature)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
=======
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import general_diseases as gd
from general_diseases import Diagnose
import special_diseases as sd
from special_diseases import Alzheimer
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import pandas as pd
import pickle
from interpret import show
import re
import io
import sys
from matplotlib import pyplot as plt
from io import BytesIO


# Capture the output of show() to extract the URL
class StreamToLogger(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = None

    def write(self, message):
        super().write(message)
        # Check if the message contains the URL pattern
        match = re.search(r'http://\S+', message)
        if match:
            self.url = match.group(0)


# Load pre-trained models
diabetes_model = pickle.load(open("Diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_disease_model.sav", "rb"))
alzheimer_model = pickle.load(open("alzheimer_model.sav", "rb"))

# Sidebar menu for user navigation
with st.sidebar:
    option = option_menu("Health Assistant", 
                         ["General", "Alzheimer", "Diabetes", "Heart Disease"], 
                         default_index=0)

# General Disease Section
if option == "General":
    st.title("General Disease Prediction")
    available_diseases = gd.find_diseases.diseases()

    chosen_disease = st.selectbox("Select a disease to check:", available_diseases)
    st.write(f"Diagnosing {chosen_disease}:")
    
    symptoms = Diagnose.diagnose(chosen_disease)
    st.write("For each symptom, enter 1 if you have it, otherwise enter 0.")
    
    user_input = []
    for symptom in symptoms:
        user_input.append(st.text_input(symptom))
    
    if st.button("Submit"):
        input_data = [int(i) for i in user_input]
        result = Diagnose.Prediction(chosen_disease, symptoms, input_data)
        
        if result == 1:
            st.warning(f"Alert: You may be prone to {chosen_disease}.")
        else:
            st.success(f"Good news! You are not at risk for {chosen_disease}.")

# Alzheimer Section
elif option == "Alzheimer":
    st.title("Alzheimer Disease Prediction")

    info = pd.DataFrame({
        "Feature": ["Sex", "Age", "Education", "SES", "CDR", "MMSE", "ETIV", "NWBV", "ASF"],
        "Description": ["Gender", "Age in years", "Years of education", "Socioeconomic Status", 
                        "Clinical Dementia Rating", "Mini Mental State Exam", 
                        "Estimated Total Intracranial Volume", "Normalized Whole Brain Volume", 
                        "Atlas Scaling Factor"]
    })
    st.table(info)

    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.text_input('Sex')
        ses = st.text_input('Socioeconomic Status')
        etiv = st.text_input('Estimated Total Intracranial Volume')

    with col2:
        age = st.text_input('Age')
        cdr = st.text_input('Clinical Dementia Rating')
        nwbv = st.text_input('Normalized Whole Brain Volume')

    with col3:
        educ = st.text_input('Years of Education')
        mmse = st.text_input('Mini Mental State Exam')
        asf = st.text_input('Atlas Scaling Factor')

    if st.button("Alzheimer Test Result"):
        input_data = [sex, age, educ, ses, mmse, cdr, etiv, nwbv, asf]
        
        # input_data = ['52', '1', '3', '145', '233', '1', '0', '150', '0']
        # input_data = [i for i in input_data]
        #                  OR
        input_data = [eval(i) for i in input_data]
        
        # Make prediction
        prediction = alzheimer_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')
        
        # Redirect stdout to capture output from show
        logger = StreamToLogger()
        sys.stdout = logger


        explainer = alzheimer_model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
        show(explainer) # This opens the explanation in a browser tab

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the extracted URL
        iframe_url = logger.url

        # Embed the generated URL in an iframe if available
        if iframe_url:
            st.components.v1.iframe(src=iframe_url, width=600, height=800)
        else:
            st.write("No URL found for the decision tree explanation.")


    if st.button("Show statistics"):
        avg_stats = sd.Alzheimer.alzheimer()
        user_data = [int(sex), int(age), int(educ), int(ses), int(mmse), int(cdr), int(etiv), int(nwbv), int(asf)]
        features = ["Age", "Education", "SES", "MMSE", "CDR", "ETIV", "NWBV", "ASF"]

        for i, feature in enumerate(features):
            plt.figure()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Average", "User"], [avg_stats[i], user_data[i]], color=["#902fed", "yellow"])
            ax.set_title(feature)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

# Diabetes Section
elif option == "Diabetes":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.text_input('Number of Pregnancies')
        skin_thickness = st.text_input('Skin Thickness')

    with col2:
        glucose = st.text_input('Glucose Level')
        insulin = st.text_input('Insulin Level')

    with col3:
        blood_pressure = st.text_input('Blood Pressure')
        bmi = st.text_input('BMI')

    with col1:
        dpf = st.text_input('Diabetes Pedigree Function')
        age = st.text_input('Age')

    if st.button("Diabetes Test Result"):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        
        input_data = [eval(i) for i in input_data]
        
        # Make prediction
        prediction = diabetes_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')
        
        # Redirect stdout to capture output from show
        logger = StreamToLogger()
        sys.stdout = logger


        explainer = diabetes_model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
        show(explainer) # This opens the explanation in a browser tab

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the extracted URL
        iframe_url = logger.url

        # Embed the generated URL in an iframe if available
        if iframe_url:
            st.components.v1.iframe(src=iframe_url, width=600, height=800)
        else:
            st.write("No URL found for the decision tree explanation.")


    if st.button("Show statistics"):
        avg_stats = sd.Diabetes.diabetes()
        user_data = [int(pregnancies), int(glucose), int(blood_pressure), int(skin_thickness), int(insulin), int(bmi), int(dpf), int(age)]
        features = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]

        for i, feature in enumerate(features):
            plt.figure()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Average", "User"], [avg_stats[i], user_data[i]], color=["#902fed", "yellow"])
            ax.set_title(feature)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

# Heart Disease Section
elif option == "Heart Disease":
    st.title("Heart Disease Prediction")

    info = pd.DataFrame({
        "Feature": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max HR", 
                    "Exercise Induced Angina", "ST Depression", "Peak Exercise Slope", "Major Vessels", "Thalassemia"],
        "Description": ["Age in years", "Sex", "Type of chest pain", "Resting blood pressure", "Cholesterol in mg/dl", 
                        "Fasting blood sugar > 120 mg/dl", "Resting electrocardiographic results", 
                        "Maximum heart rate achieved", "Exercise-induced angina", "ST depression", 
                        "Slope of peak exercise ST segment", "Number of major vessels colored by fluoroscopy", 
                        "Thalassemia: 0=Normal, 1=Fixed defect, 2=Reversible defect"]
    })
    st.table(info)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting BP')

    with col2:
        sex = st.text_input('Sex')
        chol = st.text_input('Cholesterol')

    with col3:
        cp = st.text_input('Chest Pain Type')
        fbs = st.text_input('Fasting Blood Sugar')

    with col1:
        restecg = st.text_input('Resting ECG')
        thalach = st.text_input('Max HR')

    with col2:
        exang = st.text_input('Exercise Induced Angina')
        oldpeak = st.text_input('ST Depression')

    with col3:
        slope = st.text_input('Slope of Peak Exercise ST Segment')
        ca = st.text_input('Number of Major Vessels')
        thal = st.text_input('Thalassemia')

    
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Prediction and result display
    if st.button('Heart Disease Test Result'):
        # input_data = ['52', '1', '3', '145', '233', '1', '0', '150', '0', '2.3', '1','0', '2']
        # input_data = [i for i in input_data]
        #                  OR
        input_data = [eval(i) for i in input_data]
        
        # Make prediction
        prediction = heart_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')
        
        # Redirect stdout to capture output from show
        logger = StreamToLogger()
        sys.stdout = logger


        explainer = heart_model.explain_global(name='Global Tree Explanation')  # Pass current input and prediction
        show(explainer) # This opens the explanation in a browser tab

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the extracted URL
        iframe_url = logger.url

        # Embed the generated URL in an iframe if available
        if iframe_url:
            st.components.v1.iframe(src=iframe_url, width=600, height=800)
        else:
            st.write("No URL found for the decision tree explanation.")

    if st.button("Show statistics"):
        avg_stats = sd.Heart.heart()
        user_data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), int(oldpeak), int(slope), int(ca), int(thal)]
        features = ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max HR", "Exercise Angina", "ST Depression", "Slope", "Major Vessels", "Thalassemia"]

        for i, feature in enumerate(features):
            plt.figure()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Average", "User"], [avg_stats[i], user_data[i]], color=["#902fed", "yellow"])
            ax.set_title(feature)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
>>>>>>> 49c7fde11ccd284ef92fcfd1b93ce1c721e992e2
