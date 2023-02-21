import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer
from sklearn.cluster import Birch
import statistics as sts
import streamlit as st


st.set_page_config(layout="wide")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-vector/high-resolution-world-map-with-continent-different-color-high-detail-world-map_601298-77.jpg?w=826");
             background-attachment: fixed;
	     background-position: 25% 75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title('Model Deployment: Clustering')
st.sidebar.header('Input Country listed')

input = st.sidebar.selectbox("Select Country 1 from list",('NIFTY','BANKNIFTY','3MINDIA','AARTIDRUGS','AARTIIND','AAVAS','ABB','ABCAPITAL','ABFRL','ACC',
							                                               'ACCELYA','ADANIENT','ADANIGAS','ADANIGREEN','ADANIPORTS','ADANIPOWER','ADANITRANS','ADVENZYMES',
                                                             'AEGISCHEM','AFFLE','AHLUCONT','AIAENG','AJANTPHARM','AKZOINDIA','ALKEM','ALKYLAMINE','ALLCARGO',
                                                             'AMARAJABAT','AMBER','AMBUJACEM','APARINDS','APLAPOLLO','APLLTD','APOLLOHOSP','APOLLOTYRE','ARVINDFASN',
                                                             'ASAHIINDIA','ASHOKA','ASHOKLEY','ASIANPAINT','ASTERDM','ASTRAL','ASTRAZEN','ATUL','AUBANK','AUROPHARMA',
                                                             'AVANTIFEED','AXISBANK','BAJAJ-AUTO','BAJAJCON','BAJAJELEC','BAJAJFINSV','BAJAJHLDNG','BAJFINANCE',
                                                             'BALKRISIND','BALMLAWRIE','BALRAMCHIN','BANDHANBNK','BANKBARODA','BANKINDIA','BASF','BATAINDIA',
                                                             'BBTC','BDL','BEL','BEML','BERGEPAINT','BHARATFORG','BHARATRAS','BHARTIARTL','BHEL','BIOCON',
                                                             'BIRLACORPN','BLUEDART','BLUESTARCO','BOSCHLTD','BPCL','BRIGADE','BRITANNIA','BSE','BSOFT','CADILAHC',
                                                             'CANBK','CANFINHOME','CAPLIPOINT','CARBORUNIV','CASTROLIND','CCL','CDSL','CEATLTD','CENTRALBK',
                                                             'CENTURYPLY','CENTURYTEX','CERA','CESC','CGCL','CHALET','CHAMBLFERT','CHOLAFIN','CHOLAHLDNG','CIPLA',
                                                             'COALINDIA','COCHINSHIP','COLPAL','CONCOR','COROMANDEL','CREDITACC','CRISIL','CROMPTON','CSBBANK',
                                                             'CUB','CUMMINSIND','CYIENT','DABUR','DALBHARAT','DBCORP','DBL','DCBBANK','DCMSHRIRAM','DEEPAKNTR',
                                                             'DELTACORP','DEN','DHANUKA','DIAMONDYD','DIVISLAB','DIXON','DLF','DMART','DRREDDY','ECLERX','EDELWEISS',
                                                             'EICHERMOT','EIDPARRY','EIHOTEL','ELGIEQUIP','EMAMILTD','ENDURANCE','ENGINERSIN','EQUITAS','ERIS',
							                                               'ESABINDIA','ESCORTS','ESSELPACK','EXIDEIND','FACT','FAIRCHEM','FCONSUMER','FDC','FEDERALBNK')
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
