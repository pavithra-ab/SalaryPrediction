import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('salary_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_coun = data["le_country"]
le_edu = data["le_education"]
le_remote = data["le_remote"]
mm = data["scaler"]


def show_predict_page():

    st.title("Software Developer Salary Prediction")

    st.write("We need some information to predict the salary")

    countries = (
        'United States of America',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'Canada', 'India',
        'France','Netherlands',
        'Australia','Brazil',
        'Spain','Sweden','Italy',
        'Poland','Switzerland',
        'Denmark','Norway','Israel',
        'Other'
    )
    
    education = (
                  'Less than a Bachelors', 
                  'Bachelor’s degree',
                  'Master’s degree',
                   'Professional degree'

                 )
    remote_work = ('Remote', 
                   'Hybrid (some remote, some in-person)',
                     'In-person')
    
    age = ('Under 18 years old',
           '18-24 years old',
           '25-34 years old',
            '35-44 years old',
            '45-54 years old',
            '55-64 years old', 
            '65 years or older',
             'Prefer not to say')
    
    country = st.selectbox("Country",countries)
    education = st.selectbox("Education",education)
    remotework = st.selectbox("WorkType",remote_work)
    age = st.selectbox("Age",age)
    experience = st.slider("Years Of Experience", 0,50,3)

    ok = st.button("Calculate Salary")

    if ok:
        
        if age == 'Under 18 years old':
            age_inp = 17
        if age == '18-24 years old':
            age_inp = 18
        if age == '25-34 years old':
            age_inp = 25
        if age == '35-44 years old':
            age_inp = 35
        if age == '45-54 years old':
            age_inp = 45
        if age == '55-64 years old':
            age_inp = 55
        if age == '65 years or older':
            age_inp = 65
        if age == 'Prefer not to say':
            age_inp = 0

        X = np.array([[country,age_inp,remotework,education,experience]])

        X[:,0] = le_coun.transform(X[:,0])
        X[:,2] = le_remote.transform(X[:,2])
        X[:,3] = le_edu.transform(X[:,3])
        X = X.astype(float)
        X = mm.transform(X)

        salary = regressor.predict(X)
        st.subheader(f"The estimated Annual salary is ${salary[0]:.2f}")
        


        
show_predict_page()




