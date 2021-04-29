import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/chaitu0018/data/main/Salary_Data.csv")
x = data.iloc[:,0:1].values
y = data.iloc[:,1].values
lr = LinearRegression()
lr.fit(x,y)

nav = st.sidebar.radio("Navigation",["Home","Prediction"])
if(nav=="Home"):
    st.title('Salary Predictor')
    st.image('https://4d0q7u11k31p1qnkosal8q2e-wpengine.netdna-ssl.com/wp-content/uploads/2017/04/Salary-Stacks-Up-818x300-720x300.png')
    if st.checkbox("Show data"):
        st.table(data)
    if st.checkbox('Show Graph'):
        k = plt.figure(figsize=(10,5))
        k = plt.scatter(data['YearsExperience'],data['Salary'])
        k = plt.xlabel("Years of experience")
        k = plt.ylabel("Salary")
        k = plt.tight_layout()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(k)
if(nav=="Prediction"):
    st.title("Experience vs Salary")
    st.image("https://image.freepik.com/free-vector/flat-thinking-concept_23-2148163823.jpg",width=500,height=400)
    st.header("Know your salary")
    val = st.number_input("Enter your Years of experirnce",0.00,20.00,step=0.25)
    pred = lr.predict([[val]])
    a=pred[0]
    a = "{:.2f}".format(a)
    if st.button("Predict"):
        st.success(f"Your predicted salary is {a}")