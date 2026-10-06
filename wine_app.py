import streamlit as st
import numpy as np
import joblib

# Load the saved model
obj = joblib.load('wine_model.joblib')

# Extract model and columns
model = obj['model']
cols = obj['columns']

st.title('Wine.app')

Input=[]
for i in cols:
    v=st.number_input(f'Enter {i} value :')
    Input.append(v)
if st.button('click'):
    out=model.predict([Input])
    st.success(f'The wine quality is : {out}')

