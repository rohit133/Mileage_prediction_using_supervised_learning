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
x=dataset['displacement'].values.reshape(-1,1)
y=dataset['mpg'].values.reshape(-1,1)

plt.scatter(x,y)
xd_tr,xd_ts,yd_tr,yd_ts=train_test_split(x,y)

reg=LinearRegression()
reg.fit(xd_tr, yd_tr)
y_pred=reg.predict(xd_ts)

r2=r2_score(yd_ts, y_pred)
rmse=np.sqrt(mean_squared_error(yd_ts, y_pred))
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)
plt.scatter(xd_ts,yd_ts)
plt.plot(xd_ts,y_pred,color='brown') 
plt.xlabel('Displacement')
plt.ylabel('Miles per gallon')
plt.show()


#polynomial Reggression 
poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x)
xdp_tr,xdp_ts,ydp_tr,ydp_ts=train_test_split(x_poly,y)
lin=LinearRegression()
lin.fit(xdp_tr, ydp_tr)
y_pred_poly=lin.predict(xdp_ts)
r2=r2_score(ydp_ts, y_pred_poly)
rmse=np.sqrt(mean_squared_error(yd_ts, y_pred_poly))
print('Polynomial Regression')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)
plt.scatter(xd_ts,yd_ts)
plt.plot(xd_ts,y_pred_poly,color='blue') 
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')
plt.show()


