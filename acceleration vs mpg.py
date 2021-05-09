import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
#loading the dataset 
dataset=pd.read_csv("auto-mpg.csv")
#printing the columns and descrie
print(dataset.columns)
print(dataset.describe())
##finding inconsistence 
print(dataset.info())

x=dataset['acceleration'].values.reshape(-1,1)
y=dataset['mpg'].values.reshape(-1,1)
plt.scatter(x,y)

xa_tr,xa_ts,ya_tr,ya_ts=train_test_split(x,y)

reg=LinearRegression()
reg.fit(xa_tr, ya_tr)
y_pred=reg.predict(xa_ts)

r2=r2_score(ya_ts, y_pred)
rmse=np.sqrt(mean_squared_error(ya_ts, y_pred))
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)

plt.scatter(xa_ts,ya_ts)
plt.plot(xa_ts,y_pred,color='red') 
plt.xlabel('Acceleration')
plt.ylabel('Miles per gallon')
plt.show()

#polynomial reggression 
poly = PolynomialFeatures(degree = 3) 
x_poly = poly.fit_transform(x) 
xap_tr,xap_ts,yap_tr,yap_ts=train_test_split(x_poly,y) 
lin=LinearRegression()
lin.fit(xap_tr, yap_tr)
y_pred_poly=lin.predict(xap_ts)
r2=r2_score(yap_ts, y_pred_poly)
rmse=np.sqrt(mean_squared_error(ya_ts, y_pred_poly))
print('Polynomial Regression')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)

plt.scatter(xa_ts,ya_ts)
plt.plot(xa_ts,y_pred_poly,color='blue') 
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')
plt.show()
