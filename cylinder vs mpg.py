import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
#loading the dataset 
dataset=pd.read_csv('auto-mpg.csv')
##printing the columns and descrie
print(dataset.columns)
#print(dataset.describe())
##finding inconsistence 
#print(dataset.info())

x=dataset['cylinders'].values.reshape(-1,1)
y=dataset['mpg'].values.reshape(-1,1)
plt.scatter(x,y)

xc_tr,xc_ts,yc_tr,yc_ts=train_test_split(x,y)

reg=LinearRegression()
reg.fit(xc_tr, yc_tr)

y_pred=reg.predict(xc_ts)
r2=r2_score(yc_ts, y_pred)
rmse=np.sqrt(mean_squared_error(yc_ts, y_pred))
print('R squared:',r2)
print('Root_Mean_Squared_error:',rmse)

plt.scatter(xc_ts,yc_ts)
plt.plot(xc_ts,y_pred,color='blue') 
plt.xlabel('Cylinde')
plt.ylabel('Miles per gallon')
plt.show()


#polynomial reggression 
poly = PolynomialFeatures(degree = 3) 
x_poly = poly.fit_transform(x) 
xcp_tr,xcp_ts,ycp_tr,ycp_ts=train_test_split(x_poly,y) 
lin=LinearRegression()
lin.fit(xcp_tr, ycp_tr)

y_pred_poly=lin.predict(xcp_ts)

r2=r2_score(ycp_ts, y_pred_poly)
rmse=np.sqrt(mean_squared_error(yc_ts, y_pred_poly))
print('Polynomial Regression')
print('Root squared:',r2)
print('Root_Mean_Squared_error:',rmse)


plt.scatter(xc_ts,yc_ts)
plt.plot(xc_ts,y_pred_poly,color='blue') 
plt.xlabel('HorsePower')
plt.ylabel('Miles per gallon')
plt.show()