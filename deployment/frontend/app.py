import streamlit as st
import requests
import json
import pandas as pd
from PIL import Image
import pickle
import sklearn

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="💻",
    initial_sidebar_state="collapsed",
    layout = "wide",
    menu_items={
        'Get Help': 'https://www.google.com/',
        'Report a bug': "https://github.com/marwanmusa",
        'About': "# Milestone 1 - Churn Prediction Application"
    }
)

col1, col2, col3 = st.columns([2.5, 7.5, 2.5])
with col2:
    image = Image.open('logo.png')
    st.image(image, use_column_width=True, width=800)

st.markdown(" --- ")
col1, col2 = st.columns(spec = 2)
with col1:

    image = Image.open('demographic.png')
    st.image(image, use_column_width=True, width=300)


with col2:
    with st.container():
        labels1 = ['Female', 'Male']
        gender = st.selectbox("Gender", labels1, help = "The customer’s gender")
        
    with st.container():
        labels2 = ['No', 'Yes']
        SenCit = st.selectbox("Is the customer age 65 years or older?", labels2, help = "Indicates if the customer is 65 or older")
        if SenCit == 'No':
            SeniorCitizen = 0
        else:
            SeniorCitizen = 1
            
    with st.container():
        labels2 = ['No', 'Yes']
        Partner = st.selectbox("Partner", labels2, help = "Does the customer have a partner or not?")

    with st.container():
        labels2 = ['No', 'Yes']
        Dependents = st.selectbox("Dependents", labels2, help = "Indicates if the customer lives with any dependents. Dependents could be children, parents, grandparents, etc.")

st.markdown(" ")

st.markdown(" --- ")
st.markdown(" ")


cola, colb = st.columns(2)
with cola:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")


    image = Image.open('services.png')
    st.image(image, use_column_width=True, width=400)

with colb:
    with st.container():
        labels2 = ['No phone service', 'No', 'Yes']
        MultipleLines = st.selectbox("Multiple Lines:", labels2, help = "Indicates if the customer subscribes to multiple telephone lines with the company")

    with st.container():
        labelsis = ['DSL', 'Fiber optic', 'No']
        InternetService = st.selectbox("Internet Service:", labelsis, help = "Indicates if the customer subscribes to Internet service with the company")

    with st.container():
        labels2 = ['No', 'Yes', 'No internet service']
        OnlineSecurity = st.selectbox("Online Security:", labels2, help = "Indicates if the customer subscribes to an additional online security service provided by the company")
    
    with st.container():
        labels2 = ['No', 'Yes', 'No internet service']
        OnlineBackup = st.selectbox("Online Backup:", labels2, help = "Indicates if the customer subscribes to an additional online backup service provided by the company")

    with st.container():
        labels2 = ['No', 'Yes', 'No internet service']
        DeviceProtection = st.selectbox("Device Protection:", labels2, help = "Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company")

    with st.container():
        labels2 = ['No', 'Yes', 'No internet service']
        TechSupport = st.selectbox("Tech Support:", labels2, help = "Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times")
    
    with st.container():
        labels2 = ['No', 'Yes', 'No internet service']
        StreamingTV = st.selectbox("Streaming TV:", labels2, help = "Indicates if the customer uses their Internet service to stream television programing from a third party provider")

    with st.container():
        labels2 = ['No', 'Yes', 'No internet service']
        StreamingMovies = st.selectbox("Streaming Movies:", labels2, help = "Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company")

    with st.container():
        labelscon = ['Month-to-month', 'One year', 'Two year']
        Contract = st.selectbox("Contract:", labelscon, help = "Indicates the customer’s current contract type")
        
    with st.container():
        labels2 = ['No', 'Yes']
        PaperlessBilling = st.selectbox("Paperless Billing:", labels2, help = "Indicates if the customer has chosen paperless billing")

    with st.container():
        labelspm = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        PaymentMethod = st.selectbox("Payment Method:", labelspm, help = "Indicates how the customer pays their bill")

    with st.container():
        MonthlyCharges = st.number_input("Monthly Charges:", value = 50, help = "Indicates the customer’s current total monthly charge for all their services from the company")
    
    with st.container():
        tenure = st.number_input("Tenure:", value = 24, help = "Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.")
st.markdown(" --- ")


isidata = [gender, SeniorCitizen, Partner, Dependents, tenure,
       MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
       DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
       Contract, PaperlessBilling, PaymentMethod, MonthlyCharges]       
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']

data_ = pd.DataFrame(data = [isidata], columns = columns)    

with open("scale_n_encode.pkl", 'rb') as f:
        scale_n_encode = pickle.load(f)

new_datax = scale_n_encode.transform(data_)
new_data = new_datax.tolist()

input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": new_data,
})

URL = 'http://tf-serving-churnapp.herokuapp.com/v1/models/churn:predict'

response = requests.post(URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()

st.markdown("<h2 style='text-align: center; color: black;'>Customer's Data Recap</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    st.write(data_)
    
    submitted = st.button("Submit Data")

    if submitted:
            for res in response['predictions'][0]:
                if res > 0.5:
                    st.title("The Customer will leave")
                else:
                    st.title("The Customer will stay")

