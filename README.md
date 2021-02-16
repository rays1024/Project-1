# Advanced Applied Machine Learning Project 1

## Dataset Description and Exploration

## Imports

## Linear Regression
![alt text](https://github.com/rays1024/Project-1/blob/main/[image name]?raw=true)


## Kernel Weighted Local Regression
### Gaussian Kernel
```
yhat_lk = model_lowess(dat_train,dat_test,Gaussian,0.25)
mae_lk = mean_absolute_error(dat_test[:,1], yhat_lk)
print("MAE Kernel Weighted Regression = ${:,.2f}".format(1000*mae_lk))

MAE Kernel Weighted Regression = $3,763.59
```
### Tricubic Kernel
```
yhat_lk = model_lowess(dat_train,dat_test,tricubic,0.68)
mae_lk = mean_absolute_error(dat_test[:,1], yhat_lk)
print("MAE Kernel Weighted Regression = ${:,.2f}".format(1000*mae_lk))

MAE Kernel Weighted Regression = $3,753.22
```
### Epanechnikov Kernel
```
yhat_lk = model_lowess(dat_train,dat_test,Epanechnikov,0.53)
mae_lk = mean_absolute_error(dat_test[:,1], yhat_lk)
print("MAE Kernel Weighted Regression = ${:,.2f}".format(1000*mae_lk))

MAE Kernel Weighted Regression = $3,750.64
```
### Quartic Kernel
```
yhat_lk = model_lowess(dat_train,dat_test,Quartic,0.68)
mae_lk = mean_absolute_error(dat_test[:,1], yhat_lk)
print("MAE Kernel Weighted Regression = ${:,.2f}".format(1000*mae_lk))

MAE Kernel Weighted Regression = $3,759.42
```
## Neural Networks

## XGboost

## SVR 
