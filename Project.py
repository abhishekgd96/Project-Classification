import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer
from sklearn.cluster import Birch
import statistics as sts
import streamlit as st

input = 'Norway'

#Model Prediction - Deployment

data = pd.read_csv("https://raw.githubusercontent.com/abhishekgd96/Project-Classification/main/World_development_mesurement.csv")

data['Business Tax Rate'] = data['Business Tax Rate'].str.replace('%', '')
data['Business Tax Rate'] = data['Business Tax Rate'].astype(float)

data['GDP'] = data['GDP'].str.replace('$', '')
data['GDP'] = data['GDP'].str.replace(',', '')
data['GDP'] = data['GDP'].astype(float)

data['Health Exp/Capita'] = data['Health Exp/Capita'].str.replace('$', '')
data['Health Exp/Capita'] = data['Health Exp/Capita'].str.replace(',', '')
data['Health Exp/Capita'] = data['Health Exp/Capita'].astype(float)

data['Tourism Inbound'] = data['Tourism Inbound'].str.replace('$', '')
data['Tourism Inbound'] = data['Tourism Inbound'].str.replace(',', '')
data['Tourism Inbound'] = data['Tourism Inbound'].astype(float)

data['Tourism Outbound'] = data['Tourism Outbound'].str.replace('$', '')
data['Tourism Outbound'] = data['Tourism Outbound'].str.replace(',', '')
data['Tourism Outbound'] = data['Tourism Outbound'].astype(float)




data1 = data.drop('Country', axis=1)


# calling the  MICE class
mice_imputer = IterativeImputer()
# imputing the missing value with mice imputer
data1 = mice_imputer.fit_transform(data1)


df = pd.DataFrame(data1, columns = ['Birth Rate',
                                    'Business Tax Rate',
                                    'CO2 Emissions',
                                    'Days to Start Business',
                                    'Ease of Business',
                                    'Energy Usage',
                                    'GDP',
                                    'Health Exp % GDP',
                                    'Health Exp/Capita',
                                    'Hours to do Tax',
                                    'Infant Mortality Rate',
                                    'Internet Usage',
                                    'Lending Interest',
                                    'Life Expectancy Female',
                                    'Life Expectancy Male',
                                    'Mobile Phone Usage',
                                    'Number of Records',
                                    'Population 0-14',
                                    'Population 15-64',
                                    'Population 65+',
                                    'Population Total',
                                    'Population Urban',
                                    'Tourism Inbound',
                                    'Tourism Outbound'])

scaler = StandardScaler()
df_std = scaler.fit_transform(df)

df['Country'] = data['Country']
df = df[ ['Country'] + [ col for col in df.columns if col != 'Country' ] ]
df_country =df[df['Country']==input]

df_country = df_country.drop(["Country"], axis = 1)

#Creating a BIRCH model 
model = Birch(branching_factor = 50, n_clusters = 4, threshold = 1.5)
model.fit(df_std)

pred = model.predict(df_country)
pred = sts.mode(pred)
pred
