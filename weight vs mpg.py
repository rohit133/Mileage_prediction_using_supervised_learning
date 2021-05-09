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
##finding inconsistence 
print(dataset.info())
##Resovling the errors
##create a static for weight 
x=dataset['weight'].values.reshape(-1,1)
y=dataset['mpg'].values.reshape(-1,1)
plt.scatter(x,y)


xw_tr,xw_ts,yw_tr,yw_ts=train_test_split(x,y)

reg=LinearRegression()
reg.fit(xw_tr, yw_tr)
y_pred=reg.predict(xw_ts)

r2=r2_score(yw_ts, y_pred)
rmse=np.sqrt(mean_squared_error(yw_ts, y_pred))
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)
plt.scatter(xw_ts,yw_ts)
plt.plot(xw_ts,y_pred,color='black') 
plt.xlabel('Weight')
plt.ylabel('Miles per gallon')
plt.show()

#polynomial Reggression 
poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x)
xwp_tr,xwp_ts,ywp_tr,ywp_ts=train_test_split(x_poly,y)
lin=LinearRegression()
lin.fit(xwp_tr, ywp_tr)
y_pred_poly=lin.predict(xwp_ts)
r2=r2_score(ywp_ts, y_pred_poly)
rmse=np.sqrt(mean_squared_error(yw_ts, y_pred_poly))
print('Polynomial Regression')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)

plt.scatter(xw_ts,yw_ts)
plt.plot(xw_ts,y_pred_poly,color='green') 
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')
plt.show()
