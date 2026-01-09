import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
#from model_training import rfr, le_X, sc


st.title('Horse Race Prediction')
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Horse_race_model_1.joblib"


data=joblib.load(MODEL_PATH)

le=data['LabelEncoder']
model = data['model']
sc = data['StandardScaler']    

st.success('Model Successfully Loaded')

st.header('Enter Patient data')

with st.form("Horse_Race_Form"):

    Horse_Name = st.text_input("Horse Name")
    
    Horse_Age = st.number_input("Horse Age", 3, 14, 5)
    
    Race_Year = st.number_input("Race Year", 2010, 2024, 2020)
    
    Rider_Name = st.text_input("Rider Name")
    
    Trainer_Name = st.text_input("Trainer Name")
    
    Race_Location = st.selectbox("Race Location", ["Australia", "UK", "France", "Hong Kong", "UAE", "Japan","USA"])
    
    Weather_Condition = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Windy", "Overcast", "Rainy"] )
    
    Horse_Breed = st.selectbox("Horse Breed", ["Arabian", "Quarter Horse", "Thoroughbred", "Standardbred"])
    
    Race_Distance = st.number_input("Race Distance (m)", 800, 3199, 1500)
    
    Race_Finishing_Position = st.number_input("Race Finishing Position", 1, 19, 5)

    Horse_Speed = st.number_input("Horse Speed (km/h)", 45, 75, 50)
    
    Total_Wins = st.number_input("Total Wins", 0, 39, 23)
    
    Horse_Height = st.number_input("Horse Height (cm)", 135, 175, 160)
    
    Horse_Weight = st.number_input("Horse Weight (kg)", 350, 599, 500)
    
    Track_Condition_Score = st.number_input("Track Condition Score (1-10)", 1, 10, 5)
    
    Experience_Level = st.number_input("Experience Level (1-5)", 1, 5, 3)
    
    Jockey_Win_Rate = st.number_input("Jockey Win Rate (%)", 1.0, 40.0, 25.0)
    
    Horse_Injury_Risk = st.number_input("Horse Injury Risk Score (1-10)", 1, 10, 3)
    
    Stamina_Score = st.number_input("Stamina Score (1-100)", 1, 100, 50)
    
    Acceleration_Rating = st.number_input("Acceleration Rating (1-100)", 30, 99, 60)
    
    Prize_Money = st.number_input("Prize Money ($)", 503, 499998, 50000)

    submitted = st.form_submit_button("Predict Next Race Win %")

if submitted:
    
    input_df = pd.DataFrame([{
        "Horse Name": Horse_Name,
        "Horse Age": Horse_Age,
        "Race Year": Race_Year,
        "Rider Name": Rider_Name,
        "Trainer Name": Trainer_Name,
        "Race Location": Race_Location,
        "Weather Condition": Weather_Condition,
        "Horse Breed": Horse_Breed,
        "Race Distance (m)": Race_Distance,
        "Race Finishing Position": Race_Finishing_Position,
        "Horse Speed (km/h)": Horse_Speed,
        "Total Wins":Total_Wins,
        "Horse Height (cm)": Horse_Height,
        "Horse Weight (kg)": Horse_Weight,
        "Track Condition Score (1-10)": Track_Condition_Score,
        "Experience Level (1-5)": Experience_Level,
        "Jockey Win Rate (%)": Jockey_Win_Rate,
        "Horse Injury Risk Score (1-10)": Horse_Injury_Risk,
        "Stamina Score (1-100)": Stamina_Score,
        "Acceleration Rating (1-100)": Acceleration_Rating,
        "Prize Money ($)": Prize_Money
    }])


    for col in input_df.select_dtypes('object'):
        input_df[col] = le[col].transform(input_df[col])
        print(le[col])
    # input_df[input_df.columns]= sc.transform(input_df.columns)
    
    input_df = pd.DataFrame(sc.transform(input_df), columns=input_df.columns)


    pred = model.predict(input_df)[0]

    st.subheader('Prediction Result')

    if pred < 50.00 :
        st.error(f'Horse Winning Percentage {pred:.2f}%')
    else :
        st.success(f'Horse Winning Percentage {pred:.2f}%')

