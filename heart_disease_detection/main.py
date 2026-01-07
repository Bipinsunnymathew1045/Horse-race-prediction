import streamlit as st
import pandas as pd
import joblib


st.title('Heart Disease Prediction')

data=joblib.load('Heart_disease_detection_new.joblib')

label_x=data['LabelEncoder_x']
label_y = data['LabelEncoder_y']
model = data['model']
sc = data['StandardScalar']    

st.success('Model Successfully Loaded')

st.header('Enter Patient data')

with st.form('Input_Form'):
    Age=st.number_input('Age',0,120,50)
    Gender=st.selectbox('Gender',['Male','Female'])
    Weight = st.number_input('Weight',1,300,75)
    Height = st.number_input('Height',30,250,170) 
    BMI    = st.number_input('BMI',5.0,80.0,25.0)
    Smoking = st.selectbox('Smoking',['Never','Current','Former'])
    Alcohol_Intake = st.selectbox('Alcohol Intake',['Low','Moderate','High'])
    Physical_Activity = st.selectbox('Physical_Activity',['Moderate','Sedentary','Active'])
    Diet = st.selectbox('Diet',['Average','Healthy','Unhealthy'])
    Stress_Level = st.selectbox('Stress_level',['Low','Medium','High'])
    Hypertension = st.selectbox('Hypertension',[0,1]) 
    Diabetes = st.selectbox('Diabetes',[0,1])
    Hyperlipidemia = st.selectbox('Hyperlipidemia',[0,1])
    Family_History = st.selectbox('Family_History',[0,1])
    Previous_Heart_Attack = st.selectbox('Previous_Heart_Attack',[0,1])
    Systolic_BP = st.number_input('Systolic_BP',40,250,120)
    Diastolic_BP = st.number_input('Diastolic_BP',30,180,80)
    Heart_Rate = st.number_input('Heart_Rate',20,200,75)
    Blood_Sugar_Fasting = st.number_input('Blood_Sugar_Fasting',10,300,100)
    Cholesterol_Total = st.number_input('Cholesterol_Total',50,600,180)
    
    submitted =st.form_submit_button('Predict')

if submitted:
    input_df=pd.DataFrame([{
        'Age': Age,
        'Gender'  : Gender,
        'Weight' : Weight,
        'Height' : Height,
        'BMI'   : BMI,
        'Smoking' : Smoking,
        'Alcohol_Intake' : Alcohol_Intake,
        'Physical_Activity' : Physical_Activity,
        'Diet' : Diet,
        'Stress_Level' :  Stress_Level,
        'Hypertension' : Hypertension, 
        'Diabetes' : Diabetes,
        'Hyperlipidemia' :  Hyperlipidemia,
        'Family_History' : Family_History,
        'Previous_Heart_Attack' : Previous_Heart_Attack,
        'Systolic_BP' : Systolic_BP,
        'Diastolic_BP' : Diastolic_BP,
        'Heart_Rate' : Heart_Rate,
        'Blood_Sugar_Fasting' : Blood_Sugar_Fasting,
        'Cholesterol_Total' : Cholesterol_Total
    
    }])


    for col in input_df.select_dtypes('object'):
        input_df[col] = label_x[col].transform(input_df[col])

    # input_df[input_df.columns]= sc.transform(input_df.columns)
    
    input_df = pd.DataFrame(sc.transform(input_df), columns=input_df.columns)


    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader('Prediction Result')

    if pred ==1 :
        st.error(f'Heart Disease Detected (probablity: {prob[1]*100:.2f}%)')
    else :
        st.success(f'No Heart Disease  (probablity: {prob[0]*100:.2f}%)')

