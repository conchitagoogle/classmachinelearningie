import pickle
import streamlit as st
import pandas as pd
import numpy as np

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Financial Inclusion in Africa</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

# Create a subheader with subheader method
st.subheader("Get your financial inclusion status")

# Create a markdown with markdown method
st.markdown(
    """
    ## About the app

    This app predicts the financial inclusion status of an individual in Africa.

    ## About the data

    The data was collected during the Finscope survey in 2016 and 2018.

    ## About the model

    The model was trained using XGBoost Classifier.
    """
)

# Create a text with text method
st.text("Please fill the form below")

# Load the pickled model
with open('conchita-fin-inclusion-last.pkl', 'rb') as f:
    model = pickle.load(f)

# We created selectbox for categorical columns and used slider 
# numerical values ,specified range and step 

# Create a selectbox for the location
countries = ('Kenya', 'Uganda', 'Tanzania', 
'Burundi', 'Rwanda', 'South Sudan',
'Ethiopia', 'Somalia', 'Djibouti', 'Sudan', 'Eritrea',
'Central African Republic', 'Congo', 'Democratic Republic of the Congo',
'Angola', 'Mozambique', 'Malawi', 'Zambia', 'Mauritius', 'Madagascar',
'Zimbabwe', 'Namibia', 'Botswana', 'Lesotho', 'Swaziland', 'Comoros',
'Seychelles', 'Mauritania', 'Gambia', 'Guinea-Bissau', 'Guinea', 'Sierra Leone',
'Liberia', 'Cote dIvoire', 'Burkina Faso', 'Ghana', 'Togo', 'Benin', 'Niger',
'Nigeria', 'Chad', 'Cameroon', 'Equatorial Guinea', 'Gabon', 'Cabo Verde',
'Sao Tome and Principe', 'Republic of the Congo')

country = st.selectbox('country', countries)

# Create a selectbox for the location type
location_type = st.selectbox('Location Type', ('Rural', 'Urban'))

# cell phone access
cellphone_access = st.selectbox('Cell Phone Access', ('Yes', 'No'))

# Create a slider for the household size
household_size = st.slider('Household Size', 1, 20, 1)

# Create a slider for the age of the respondent
respondent_age = st.slider('Respondent Age', 16, 100, 16)

# Create a job type selectbox
job_type = st.selectbox('Job Type', ('Self employed', 'Informally employed',
'Formally employed Government', 'Remittance Dependent', 'Farming and Fishing',
'Formally employed Private', 'No Income', 'Other Income', 'Government Dependent'))

# Create a marital status selectbox
marital_status = st.selectbox('Marital Status', ('Married/Living together', 'Widowed',
'Divorced/Seperated', 'Single/Never Married', 'Dont know'))

# Gender
gender_of_respondent = st.selectbox("Gender", ("Male", "Female"))

# Relationship with head
relationship_with_head = st.selectbox("Relationship with head", ("Head of Household", "Spouse", "Child", 
"Parent", "Other relative", "Other non-relatives"))    


# in order to recieved client inputs appended these inputs (created above) 
# into dictionary as we mentioned before. And We returned into dataframe.
data = {"country": country,
    "location_type": location_type, 
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "respondent_age": respondent_age,
        "job_type": job_type,
        "marital_status": marital_status,
        "gender_of_respondent": gender_of_respondent,
        "relationship_with_head": relationship_with_head}

# Convert data into dataframe
df = pd.DataFrame.from_dict([data])

# And appended column names into column list. 
# We need columns to use with reindex method as we mentioned before.
columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
]
df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
prediction = model.predict(df)

# Submit button to predict
if st.button('Predict'):
    if prediction == 0:
        st.write('You are not financially included')
    else:
        st.write('Hello, you are financially included')

# Display probabilities
if st.button('Show probabilities'):
    probabilities = model.predict_proba(df)
    st.write(probabilities)
    st.write("o = not financially included, 1 = financially included")




