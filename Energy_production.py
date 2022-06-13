# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 16:23:07 2022

@author: Nasir
"""
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz
from sklearn.model_selection import train_test_split,KFold,cross_val_score,RepeatedKFold,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pickle import load,dump

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
    
x = df.drop(['energy_production'], axis=1)
y = df['energy_production']

# Dividing into test and train 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#-------------------------------------------------------------------------------------------------------------
#selecting best models
model_selc = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(n_estimators = 10),
             GradientBoostingRegressor()]

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state= None)
cv_results = []
cv_results_mean =[]
for ele in model_selc:
    cross_results = cross_val_score(ele, x_train, y_train, cv=kfold, scoring ='r2')
   
    cv_results.append(cross_results)
   
    cv_results_mean.append(cross_results.mean())
    print("\n MODEL: ",ele,"\nMEAN R2:",cross_results.mean())
    
    



new_df2=pd.DataFrame()
for i in df:
    upper_limit = df[i].quantile(0.99)
    lower_limit = df[i].quantile(0.01)
    new_df = df[(df[i] <= upper_limit) & (df[i] >= lower_limit)]
    new_df1=new_df[i]
    new_df2[i]=new_df1
    new_df1=pd.DataFrame()
    
df=new_df2
df = df.dropna()
for i in df :
    box_plot(i)
    scater_plot(i)
    
for i in df :
    min_max(i)
    
#model1
# RandomForestRegressor

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)
from sklearn.ensemble import RandomForestRegressor
 # create regressor object
 
regressor = RandomForestRegressor(n_estimators =50, random_state = 0)
# fit the regressor with x and y data
regressor.fit(x_train, y_train) 
# Use the forest's predict method on the test data
y_pred= regressor.predict(x_test)
y_pred
# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')
RandomForestRegressor_Accuracy=r2_score*100


# xgboost
# Model 2
import xgboost as xgb
from sklearn.metrics import mean_squared_error

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.5,
                max_depth = 3, alpha = 15, n_estimators = 10)
xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)
preds
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
r2_score = xg_reg.score(x_test,y_test)
print(r2_score*100,'%')
from sklearn.metrics import r2_score
score=r2_score(y_test,preds)
score
xgboost_Accuracy=score*100

#model 3
#DecisionTreeRegressor
# import the regressor
from sklearn.tree import DecisionTreeRegressor 
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
regressor.fit(x_train, y_train)
# predicting a new value
  
# test the output by changing values, like 3750
y_pred = regressor.predict(x_test)
  
y_pred
Dtr_score=regressor.score(x_test, y_test)
DecisionTreeRegressor_Accuracy=Dtr_score*100


print("RandomForestRegressor_Accuracy",RandomForestRegressor_Accuracy)
print("xgboost_Accuracy",xgboost_Accuracy)
print("DecisionTreeRegressor_Accuracy",DecisionTreeRegressor_Accuracy)

dump(regressor,open('RandomForestRegressor_Energy_production.pkl','wb'))
#------------------------------------------------------------------------------------------------------------------
model = load(open('RandomForestRegressor_Energy_production.pkl','rb'))

