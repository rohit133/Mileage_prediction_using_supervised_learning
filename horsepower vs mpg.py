import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 

#loading the dataset 
dataset=pd.read_csv('auto-mpg.csv')
#printing the columns and descrie
print(dataset.columns)
print(dataset.describe())
#finding inconsistence 
print(dataset.info())
#Resovling the errors
dataset['horsepower']=pd.to_numeric(dataset['horsepower'],errors='coerce')
#Importig the imputer and changing the NaN valuse to a numbric values 
from sklearn.impute import SimpleImputer
im=SimpleImputer()#create a element of the iputer 
val=im.fit_transform(dataset['horsepower'].values.reshape(-1,1))
dataset['horsepower']=val
#checking the errors
print(dataset.info())
x=dataset['horsepower'].values.reshape(-1,1)
y=dataset['mpg'].values.reshape(-1,1)
plt.scatter(x,y)
xh_tr,xh_ts,yh_tr,yh_ts=train_test_split(x,y)
reg=LinearRegression()
reg.fit(xh_tr, yh_tr)
y_pred=reg.predict(xh_ts)

r2=r2_score(yh_ts, y_pred)
rmse=np.sqrt(mean_squared_error(yh_ts, y_pred))
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)
plt.scatter(xh_ts,yh_ts)
plt.plot(xh_ts,y_pred,color='green') 
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')
plt.show()


#polynomial reggression 
poly = PolynomialFeatures(degree = 3) 
x_poly = poly.fit_transform(x) 
xhp_tr,xhp_ts,yhp_tr,yhp_ts=train_test_split(x_poly,y) 
lin=LinearRegression()
lin.fit(xhp_tr, yhp_tr)
y_pred_poly=lin.predict(xhp_ts)
r2=r2_score(yhp_ts, y_pred_poly)
rmse=np.sqrt(mean_squared_error(yh_ts, y_pred_poly))
print('Polynomial Regression')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)

plt.scatter(xh_ts,yh_ts)
plt.plot(xh_ts,y_pred_poly,color='brown') 
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')
plt.show()
