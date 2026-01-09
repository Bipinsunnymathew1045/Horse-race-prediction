import numpy as np
import pandas as pd
import joblib 


data = pd.read_csv('horse_racing_full_dataset_100k.csv')
data

data.isna().sum()

data.info()

data['Trainer Name'].value_counts()

data['Rider Name'].value_counts()

data['Race Location'].value_counts()

data['Weather Condition'].value_counts()

data['Horse Breed'].value_counts()

data.columns

data.describe()

data.iloc[40:70]

data.iloc[56]

y = data['Next Race Win %']
X =  data.drop('Next Race Win %',axis=1)

from sklearn.preprocessing import LabelEncoder
le_X = {}
for i in data.select_dtypes('object'):
    le = LabelEncoder()
    X[i]=le.fit_transform(X[i])
    le_X[i] = le

le_X

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[X.columns]=sc.fit_transform(X[X.columns])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

from sklearn.linear_model import  LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

lr.score(X_test,y_test)*100

from sklearn.metrics import r2_score
y_pred = lr.predict(X_test)
res=r2_score(y_pred,y_test)
print(f'Score : {res*100:.2f}%')

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)

rfr.score(X_test,y_test)*100

from sklearn.metrics import r2_score
y_pred = rfr.predict(X_test)
res=r2_score(y_pred,y_test)
print(f'Score :{res*100:.2f}%')

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)

dtr.score(X_test,y_test)*100

X.columns

data.info()

# def race_predict(Horse_Name_in, Horse_Age_in, Race_Year_in, Rider_Name_in, Trainer_Name_in,
#         Race_Location_in, Weather_Condition_in, Horse_Breed_in,
#         Race_Distance_in, Race_Finishing_Position_in, Horse_Speed_in,
#         Total_Wins_in, Horse_Height_in, Horse_Weight_in,
#         Track_Condition_Score_in, Experience_Level_in,
#         Jockey_Win_Rate_in,Horse_Injury_Risk_Score_in,
#         Stamina_Score_in, Acceleration_Rating_in,
#         Prize_Money_in):

#     df1 = pd.DataFrame({
#                         'Horse Name': Horse_Name_in, 
#                         'Horse Age' : Horse_Age_in, 
#                         'Race Year' : Race_Year_in,
#                         'Rider Name': Rider_Name_in, 
#                         'Trainer Name': Trainer_Name_in,
#                         'Race Location': Race_Location_in, 
#                         'Weather Condition': Weather_Condition_in,
#                         'Horse Breed': Horse_Breed_in,
#                         'Race Distance (m)': Race_Distance_in,
#                         'Race Finishing Position' : Race_Finishing_Position_in,
#                         'Horse Speed (km/h)': Horse_Speed_in,
#                         'Total Wins' :  Total_Wins_in,
#                         'Horse Height (cm)' : Horse_Height_in, 
#                         'Horse Weight (kg)': Horse_Weight_in,
#                         'Track Condition Score (1-10)' : Track_Condition_Score_in,
#                         'Experience Level (1-5)': Experience_Level_in,
#                         'Jockey Win Rate (%)': Jockey_Win_Rate_in,
#                         'Horse Injury Risk Score (1-10)': Horse_Injury_Risk_Score_in,
#                         'Stamina Score (1-100)': Stamina_Score_in, 
#                         'Acceleration Rating (1-100)':  Acceleration_Rating_in,
#                         'Prize Money ($)': Prize_Money_in
#                     },index = [0])

#     for i in df1.select_dtypes('object'):
#         df1[i] = le_X[i].transform(df1[i])

#     df1[df1.columns] = sc.transform(df1[df1.columns])

#     ypred = rfr.predict(df1) 
#     return f'Win % :{round(ypred[0],2)}%'




joblib.dump({'LabelEncoder':le_X,
'model':rfr,
'StandardScaler':sc
},'Horse_race_model_2.joblib')
print('Model Saved')

# # Load model from file 
# model_load = joblib.load('Horse_race_model_1.joblib') # joblib.load('filepath')
# model=model_load['model']
# lbl_encoder = model_load['LabelEncoder']
# sc_load = model_load['StandardScaler']

# model

# sc_load

# import sklearn
# print(sklearn.__version__)

# import numpy

# print(numpy.__version__)