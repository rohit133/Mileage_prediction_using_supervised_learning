import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
#Loading dataset
dataset=pd.read_csv('auto-mpg.csv')
print(dataset)
print(dataset.columns)

dataset['horsepower']=pd.to_numeric(dataset['horsepower'],errors='coerce')
#Importig the imputer and changing the NaN valuse to a numbric values 

im=SimpleImputer()#create a element of the iputer 
val=im.fit_transform(dataset['horsepower'].values.reshape(-1,1))
dataset['horsepower']=val
print(val)

data=dataset.drop(['origin','car name','model year'],axis=1)
print(data.columns)

x=dataset['horsepower'].values.reshape(-1,1)
y=dataset['mpg'].values.reshape(-1,1)

scaler= StandardScaler()#calling the function 
scaler.fit_transform(x,y)#fitting the values of x
x_train,x_test,y_train,y_test=train_test_split(x,y)


#performing the linear regression
lin=LinearRegression()
lin.fit(x_train,y_train)
#predicting using the values
y_pred=lin.predict(x_test)
#finding the Root Squared
r2=r2_score(y_test, y_pred)
#finding the Root_mean_Squared_error
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print('Linear Regression:')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)
print('\n')



#performing the Rnadom forest 
ran=RandomForestRegressor(n_estimators=1)
ran.fit(x_train,y_train.ravel())
y_pred=ran.predict(x_test)
#finding the root Squared for Random Forset Regression 
r2=r2_score(y_test, y_pred)
#finding the Root_mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print('Random Forest Regression:')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)
print('\n')


#performing the Rnadom forest using the n_estimators=100
ran=RandomForestRegressor(n_estimators=100)
ran.fit(x_train,y_train.ravel())
y_pred=ran.predict(x_test)

#finding the root Squared for Random Forset Regression 
r2=r2_score(y_test, y_pred)
#finding the Root_mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print('Random Forest Regression using N_estimators=100:')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)

