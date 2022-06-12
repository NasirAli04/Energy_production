# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 16:23:07 2022

@author: Nasir
"""
# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz


df=pd.read_excel(r"C:\Users\Nasir\Desktop\Data_Science\Energy_prediction\energy_production.xlsx")
df.info()



df[['temperature','exhaust_vacuum','amb_pressure','r_humidity','energy_production']]=df['temperature;exhaust_vacuum;amb_pressure;r_humidity;energy_production'].str.split(';', expand=True)
df=df.drop(['temperature;exhaust_vacuum;amb_pressure;r_humidity;energy_production'], axis=1)
df=df.astype(str).astype(float)

#my_report = sweetviz.analyze([df, "Energy_production"],target_feat='energy_production')
#my_report.show_html('Report.html')

df.isnull().sum()
df.duplicated().sum()
df=df.drop_duplicates()
df.duplicated().sum()


correaltion=df.corr()
sns.set_style(style='darkgrid')
sns.pairplot(df)


def box_plot(i):
    a=sns.boxplot(df[i])
    plt.show(a)
    
    
def scater_plot(i):
    print(i)
    x=df[i]
    y=df['energy_production']
    plt.scatter(x, y)
    plt.show()

def min_max(i):
    print("min value of ",i)
    print(df[i].min())
    print("min value of ",i)
    print(df[i].max())
    
    
    
for i in df :
    box_plot(i)
    scater_plot(i)
    
for i in df :
    min_max(i)
    
    
    
    

    