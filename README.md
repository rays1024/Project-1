# Advanced Applied Machine Learning Project 1

## Dataset Description and Exploration
![alt text](https://github.com/rays1024/Project-1/blob/main/data.png?raw=true)

The data file we use contains Boston housing prices and many other variables. For our analysis focus, we will only use the number of rooms as our independent variable in predicting the house prices in thousands of dollars. From the graph, we can see that there is a general upward trend in the plot and there is also a slight curve to it.

## Imports
Here are some import codes for our analysis.

### General Imports
```
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import pyplot
```
### Imports for Creating Neural Networks
```
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
```
### Imports for Uploading Files
```
from google.colab import files
data_to_load = files.upload()
```
```
import io
df = pd.read_csv(io.BytesIO(data_to_load['Boston Housing Prices.csv']))
```
### Imports for Calculations
```
from sklearn.model_selection import train_test_split as tts
```
```
from sklearn.metrics import mean_absolute_error
```
```
from sklearn.linear_model import LinearRegression
```
```
from sklearn.model_selection import KFold
```
### Import for Creating Extreme Boosting Algorithm
```
import xgboost as xgb
```
### Import for Creating Support Vector Regression
```
from sklearn.svm import SVR
```

## Linear Regression
Linear regression is a very commonly used tool for making predictions. It estimates each independent variable for its predicting power and the significance of its result. It calculates an estimation function for which the sum of squared differences between the observations and predictions is the least. When we only have one x, the number of rooms, the linear regression produces a very simple prediction function: a straight line. This function is a very weak learner and yields a high mean absolute error of $4,433.17 as shown below.
### Linear Regression Plot
![alt text](https://github.com/rays1024/Project-1/blob/main/lm.png?raw=true)

### Validated Linear Regression MAE
```
mae_lm = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  lm.fit(X_train.reshape(-1,1),y_train)
  yhat_lm = lm.predict(X_test.reshape(-1,1))
  mae_lm.append(mean_absolute_error(y_test, yhat_lm))
print("Validated MAE Linear Regression = ${:,.2f}".format(1000*np.mean(mae_lm)))
```
Validated MAE Linear Regression = $4,433.17

## Kernel Weighted Local Regression
The kernel weighted local regression estimates the function locally as its name indicates. There are no parameters that define a function in kernel regression. For each x value, the algorithm generates an estimation for y using the individual x's neighboring values. The number of neighbors, k, determines the variance and bias of the estimation. Higher value of k yields low variance and high bias, and vice versa. We can use different kernel functions for assigning weights based on the Euclidean distance between the x value and its neighbors. Here, we used four kernel functions to determine which yields the lowest mean absolute error. After our estimations, we found that the Gaussian kernel has the lowest mean absolute error of $4,107.38.

### Functions
```
def lowess_kern(x, y, kern, tau):

    # tau is called bandwidth K((x-x[i])/(2*tau))

    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the kernel function by using only the train data    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest
```
```
def model_lowess(dat_train,dat_test,kern,tau):
  dat_train = dat_train[np.argsort(dat_train[:, 0])]
  dat_test = dat_test[np.argsort(dat_test[:, 0])]
  Yhat_lowess = lowess_kern(dat_train[:,0],dat_train[:,1],kern,tau)
  datl = np.concatenate([dat_train[:,0].reshape(-1,1),Yhat_lowess.reshape(-1,1)], axis=1)
  f = interp1d(datl[:,0], datl[:,1],fill_value='extrapolate')
  return f(dat_test[:,0])
```
#### Tricubic Kernel
```
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
```
#### Epanechnikov Kernel
```
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
```
#### Quartic Kernel
```
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)
```
#### Gaussian Kernel
```
def Gaussian(x):
  return np.where(np.abs(x)>2,0,np.exp(-1/2*x**2))
```

### Four Kernel Regressions Summary Plot
![alt text](https://github.com/rays1024/Project-1/blob/main/lk.png?raw=true)

We have ploted the estimations of the four kernel functions and there are very small differences between them. I found the optimal tau value for each kernel and they all yield very good results.

### Validated Gaussian Kernel Regression MAE
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.25)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))
```
Validated MAE Local Kernel Regression = $4,107.38

### Validated Tricubic Kernel Regression MAE
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.25)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))
```
Validated MAE Local Kernel Regression = $4,126.14

### Validated Epanechnikov Kernel Regression MAE
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.25)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))
```
Validated MAE Local Kernel Regression = $4,114.04

### Validated Quartic Kernel Regression MAE
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Quartic,0.68)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))
```
Validated MAE Local Kernel Regression = $4,123.69

## Neural Networks
The neural networks contain many nodes or neurons. Most neural networks neurons are in layers, and neurons in each layer are connected to at least a neuron one layer before it and one layer after it. The data are passed from the neurons in the first layer to those in the last layer. All neural networks have to have an activation function. In our case, since we are performing a regression, we need a linear activation. After the algorithm generates a regression from training, we need to validate the results using k-fold validation. This method splits the dataset into k groups and train the algorithm in k iterations. Each iteration, it uses one k-1 groups for training and the last kth group is used for validation. We also generalize this method to other regressions for more reliable results. The mean absolute errors from a 3-fold validation are $4,152.78, $4,134.34 and $4,108.31 respectively.

### Neural Network Regression Plot
![alt text](https://github.com/rays1024/Project-1/blob/main/nn.png?raw=true)

Here is a plot of neural networks regression. It look a lot like the plots of kernel regression, and it shows that this is also a quite strong learner.

### Validated Neural Network Regression MAE
```
mae_nn = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
  model.fit(X_train.reshape(-1,1),y_train,validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])
  yhat_nn = model.predict(X_test.reshape(-1,1))
  mae_nn.append(mean_absolute_error(y_test, yhat_nn))
print("Validated MAE Neural Network Regression = ${:,.2f}".format(1000*np.mean(mae_nn)))
```
Validated MAE Neural Network Regression = $4,152.78 </br>
Validated MAE Neural Network Regression = $4,134.34 </br>
Validated MAE Neural Network Regression = $4,108.31

## Extreme Boosting Algorithm (XGBoost)

XGBoost is a Machine Learning algorithm based on random forest algorithm. In the boosting algorithm, decision trees alternate selection criteria that creates a dynamic selection process. In XGBoost, the algorithm uses gradient descent algorithm that minimizes error and optimizes in terms of computing resources and time. This process eliminates weak learners to improve the performance of the algorithm. The validated mean absolute error for XGBoost is $4136.63.

### Validated XGBoost MAE
```
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
mae_xgb = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model_xgb.fit(X_train.reshape(-1,1),y_train)
  yhat_xgb = model_xgb.predict(X_test.reshape(-1,1))
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression = ${:,.2f}".format(1000*np.mean(mae_xgb)))
```
Validated MAE XGBoost Regression = $4,136.63

## Support Vector Regression (SVR)

The Support Vector Regression is based on the Support Vector Machine, which is an algorithm for separating two classes of data within one vector space with a hyperplane. The SVR uses this hyperplane and decision boundaries to construct a regression line that estimates the mapping from x to y. The decision boundaries allow us to decide how much error we wish to include in our results. The validated mean absolute error from SVR is $4,130.50. 

### Validated SVR MAE
```
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)

model = svr_rbf

mae_svr = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model.fit(X_train.reshape(-1,1),y_train)
  yhat_svr = model.predict(X_test.reshape(-1,1))
  mae_svr.append(mean_absolute_error(y_test, yhat_svr))
print("Validated MAE Support Vector Regression = ${:,.2f}".format(1000*np.mean(mae_svr)))
```
Validated MAE Support Vector Regression = $4,130.50

## Performance Comparison
### From the Lowest MAE to the Highest
1. Gaussian Kernel Regression ------ $4,107.38
2. Neural Networks Regression ------ $4,108.31
3. Epanechnikov Kernel Regression ------ $4,114.04
4. Quartic Kernel Regression ------ $4,123.69
5. Tricubic Kernel Regression ------ $4,126.14
6. Support Vector Regression ------ $4,130.50
7. Neural Networks Regression ------ $4,134.34
8. XGBoost ------ $4,136.63
9. Neural Networks Regression ------ $4,152.78
10. Linear Regression ------ $4,433.17

![alt text](https://github.com/rays1024/Project-1/blob/main/ranks.png?raw=true)

This is a plot of methods and their performance for a direct comparison. We did not include the data point for linear regression because it is beyond the scope of other regressors, and including it would make the differences of performance in other regressors less obvious.

Our results show that the Gaussian kernel regression yields the best result, and all kernel regressions yield decent results comparing with othre methods. The neural networks regression yields three results from a 3-fold validation, and their average performance is not very good: two of the three results are placed after the fifth position. The SVR and XGBoost methods performed about the same as the neural networks regression. The worst performance is from linear regression, which is not surprising since it is the weakest learner among all methods compared.
