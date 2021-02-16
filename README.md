# Advanced Applied Machine Learning Project 1

## Dataset Description and Exploration

## Imports

## Linear Regression
![alt text](https://github.com/rays1024/Project-1/blob/main/[image name]?raw=true)


## Kernel Weighted Local Regression
### Gaussian Kernel
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.25)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))

Validated MAE Local Kernel Regression = $4,107.38
```
### Tricubic Kernel
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.25)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))

Validated MAE Local Kernel Regression = $4,126.14
```
### Epanechnikov Kernel
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.25)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))

Validated MAE Local Kernel Regression = $4,114.04
```
### Quartic Kernel
```
mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Quartic,0.68)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lk)))

Validated MAE Local Kernel Regression = $4,123.69
```
## Neural Networks

## XGboost

## SVR 
