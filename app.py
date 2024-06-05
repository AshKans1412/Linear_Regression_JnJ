import streamlit as st
from page1 import Linear_Regression
from page2 import linear_regression_with_intercept

#st.set_page_config(page_title="Linear Regression with Gradient Descent", layout="wide")

# Create a selectbox for navigation
page = st.selectbox('Choose a model', ['Simple Linear Regression', 'Linear Regression with Intercept'])

if page == 'Simple Linear Regression':
    Linear_Regression()
elif page == 'Linear Regression with Intercept':
    linear_regression_with_intercept()