import streamlit as st
import joblib
import numpy as np
import pandas as pd

def load_model():
    with open(r"Titanic\pipline_titanic.joblib",'rb') as file:
        data = joblib.load(file)
    return data

data = load_model()

def show_predict_page():
    st.title('Titanic Survival Predictor')

    st.write('We some some information to predict the survival')

    embarked = ["S","Q","C"]
    gender = ['Male','Female','Nah nigga kys']
    Pclass = ['3','2','1']

    emb = st.selectbox('Journey Embarked From',embarked)
    gen = st.selectbox('Gender',gender)
    pclass = st.selectbox('Class',Pclass)
    age = st.number_input("Age",step=1)
    sibsp = st.number_input("Number of Siblings Onboard",min_value=0,max_value= 3,step=1)
    parch = st.number_input("Parents or Children Onboard",min_value=0,max_value=3,step=1)
    fare = st.number_input("Fare",step=1)

    sample = pd.DataFrame(np.array([pclass,gen.lower(),age,sibsp,parch,fare,emb]).reshape(1,7))
    sample = sample.rename(columns={0: 'Pclass',1:'Sex',2:'Age',3:'SibSp',4:'Parch',5:'Fare',6:'Embarked'})
    pred = data.predict(sample)
    k = st.button("Predict")
    if k:
        if pred[0]:
            st.subheader("The person would have survived")
        else:
            st.subheader("The person would not have survived")



