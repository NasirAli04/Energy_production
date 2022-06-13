# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:38:08 2022

@author: Nasir
"""

import streamlit as st
from pickle import load,dump
import pandas as pd
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>'''


st.markdown(page_bg_img, unsafe_allow_html=True)





def main():
    st.subheader("Energy_production")
    menu=["Home",'Prediction','About']
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice=="Home":
        st.header("Welcome to Energy Production Prediction")
        image="https://media0.giphy.com/media/9floxejD0yczwuZMLJ/giphy.gif?cid=ecf05e471ne9srkmabvq4j988mvqvmy3xur1w6jcuq545ie7&rid=giphy.gif&ct=g"
        st.image(image)
        
    elif choice=="Prediction":
        Temperature=st.number_input('Enter Temperature', 5, 33)
        Exhaust_vacuum=st.number_input('Enter Exhaust_vacuum', 35, 77)
        Amb_pressure=st.number_input('Enter amb_pressure',1000, 1028)
        r_humidity=st.number_input('Enter r_humidity', 38, 100)
        data=[Temperature,Exhaust_vacuum,Amb_pressure,r_humidity]
        
        df = pd.DataFrame([data])
        df.columns =['temperature','exhaust_vacuum','amb_pressure','r_humidity']
        st.write(df)
        model = load(open('RandomForestRegressor_Energy_production.pkl','rb'))
        pred= model.predict(df)
        
        st.write(pred)
        
        
        
        
        




if __name__ == '__main__':
	main()