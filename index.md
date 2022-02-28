# Concepts and Applications of Multivariate Regression Analysis and Gradient Boosting including XGBoost

### Multivariate Regression Analysis
Multivariate Regression Analysis is a 

In general we want

𝔼(𝑦|𝑋1,𝑋2,...𝑋𝑝):=𝐹(𝑋1,𝑋2,𝑋3,...𝑋𝑝) 

where  𝐹  represents the model (regressor) we consider.

Variable Selection
We want to select only the features that are really important for our model.

If the functional input-output model is  𝑌=𝐹(𝑋1,𝑋2,𝑋3,𝑋4,𝑋5...𝑋𝑝)  then we imagine that it is very possible that only a subset of the variables  𝑋1,𝑋2,𝑋3,𝑋4,𝑋5...𝑋𝑝  are important and we need to disconsider (eliminate from the model) those that are not relevant.

Programming and algorithms are based on equations, functions and statement evaluations.

To represent variable selection in a functional way, we can think of multiplying each variable from the model by a binary weight, a weight of  0  means the feature is not important and a weight of  1  means that it is important:

𝑌=𝐹(𝑤1⋅𝑋1,𝑤2⋅𝑋2,𝑤3⋅𝑋3,𝑤4⋅𝑋4,𝑤5⋅𝑋5...𝑤𝑝⋅𝑋𝑝) 

where the weights  𝑤𝑖  are either  0  or  1. 

The vector of binary weights  𝑤=(𝑤1,𝑤2,𝑤3,...𝑤𝑝)  gives us what we call the sparsity pattern for the variable selection.

Critical Aspects
What is the simplest choice for the function  𝐹 ?
How do we perform variable selection?
How do we accomodate nonlinear relationships?


Variable Selection
In the case of multiple linear regression we have that

𝐹(𝑋1,𝑋2,...𝑋𝑝)=𝛽1𝑋1+𝛽2𝑋2+...𝛽𝑝𝑋𝑝 

and the sparsity pattern means that a subset of the  𝛽1,𝛽2,...𝛽𝑝  are equal to  0. 

So we assume

𝑌≈𝑋⋅𝛽+𝜎𝜖 

and we want the coefficients  𝛽. 

The "classical" way of solving is:

𝑋𝑡⋅𝑌≈𝑋𝑡𝑋⋅𝛽+𝜎𝑋𝑡𝜖 
so we get
𝔼(𝛽)=(𝑋𝑡𝑋)−1𝑋𝑡⋅𝔼(𝑌) 

where  𝔼(𝑌)  denotes the expected value of  𝑌. 



### Gradient Boosting
Gradient Boosting is a robust machine learning algorithm made up of Gradient descent and Boosting. The word "gradient" implies that you can have two or more derivatives of the same function. Gradient Boosting has three main components: additive model, loss function, and a weak learner. It can do regression, classification, and ranking. 

We fit an additive model (ensemble) 

<img width="74" alt="image" src="https://user-images.githubusercontent.com/98488324/155919447-92691822-9966-4be1-b301-0cb6723a1485.png">

in a forward stage-wise manner.       
In each stage, introduce a weak learner to compensate the shortcomings of existing weak learners. In Gradient Boosting, "shortcomings" are identified by gradients, which tell us how to improve our model.       


More specifically, here we assume we have an regressor  𝐹  and, for the observation  𝑥𝑖  we make the prediction  𝐹(𝑥𝑖) . To improve the predictions, we can regard  𝐹  as a "weak learner" and therefore train a decision tree (we can call it  ℎ ) where the new output is  𝑦𝑖−𝐹(𝑥𝑖) . Thus, there are increased chances that the new regressor

<img width="85" alt="image" src="https://user-images.githubusercontent.com/98488324/156001320-66ec691e-d224-4d66-baac-5385ed3ce2a4.png">
is better than the old one,  𝐹. 



### Extreme Gradient Boosting (XGBoost)
XGBoost is an abbreviation for Extreme Gradient Boost (I wrote an article that provides the gist of gradient boost here). Unlike Gradient Boost, XGBoost makes use of regularization parameters that helps against overfitting.       



## Applications with Real Data
Cars Data (output variable (y) is the mileage (MPG)): 

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
cars = pd.read_csv("drive/MyDrive/DATA410_AdvML/cars.csv")
```
<img width="234" alt="image" src="https://user-images.githubusercontent.com/98488324/153694695-0e275da1-6379-44db-af1b-92a43d0c0544.png">


Boston Housing Data (output variable (y) is the number of rooms (cmedv)): 

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("drive/MyDrive/DATA410_AdvML/Boston Housing Prices.csv")
```
<img width="1179" alt="image" src="https://user-images.githubusercontent.com/98488324/155917389-b3c4e978-eac0-4f07-8c90-961fdff87ef1.png">


#### Multivariate Regression Analysis:
Import libraries and create functions:

```python
# import libraries
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from matplotlib import pyplot
import xgboost as xgb

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2))
  
# defining the kernel local regression model
def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
  
# defining the kernel boosted lowess regression model
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  #model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
```

Apply Cars data:

```python
X = cars[['ENG','CYL','WGT']].values
y = cars['MPG'].values

scale = StandardScaler()
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)

# Nested Cross-Validation
mse_lwr = []
mse_blwr = []
mse_rf = []
mse_xgb = []
mse_nn = []
for i in range(123):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    #model_nn.fit(xtrain,ytrain,validation_split=0.3, epochs=500, batch_size=20, verbose=0, callbacks=[es])
    #yhat_nn = model_nn.predict(xtest)
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_rf.append(mse(ytest,yhat_rf))
    mse_xgb.append(mse(ytest,yhat_xgb))
    #mse_nn.append(mse(ytest,yhat_nn))
print('The Cross-validated MSE for Lowess is : '+str(np.mean(mse_lwr)))
print('The Cross-validated MSE for Boosted Lowess is : '+str(np.mean(mse_blwr)))
print('The Cross-validated MSE for Random Forest is : '+str(np.mean(mse_rf)))
print('The Cross-validated MSE for Extreme Gradient Boosting (XGBoost) is : '+str(np.mean(mse_xgb)))
```

#### Final results: 

The Cross-validated MSE for Lowess is : 17.025426125745327      
The Cross-validated MSE for Boosted Lowess is : 16.656353893436698      
The Cross-validated MSE for Random Forest is : 16.947624934702624     
The Cross-validated MSE for Extreme Gradient Boosting (XGBoost) is : 16.14075756009356     

Since we aim to minimize the crossvalidated mean square error (MSE) for the better results, I conclude that Extreme Gradient Boosting (XGBoost) achieved the better result than other regressions including Lowess, Boosted Lowess, and Random Forest. 
       
       
#### The predictions we made for the test data:

```python
fig, ax = plt.subplots(figsize=(12,9))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=df['rooms'], y=df['cmedv'],s=25)
ax.plot(X_test, lm.predict(X_test), color='red',label='Linear Regression')
ax.plot(dat_test[:,0], yhat_nn, color='lightgreen',lw=2.5,label='Neural Network')
ax.plot(dat_test[:,0], model_lowess(dat_train,dat_test,Epanechnikov,0.53), color='orange',lw=2.5,label='Kernel Weighted Regression')
ax.set_xlabel('Number of Rooms',fontsize=16,color='navy')
ax.set_ylabel('House Price (Thousands of Dollars)',fontsize=16,color='navy')
ax.set_title('Boston Housing Prices',fontsize=16,color='purple')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
plt.legend()
```


Apply Boston Housing data:

```python
from sklearn.metrics import mean_absolute_error

features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df[features])
y = np.array(df['cmedv']).reshape(-1,1)
dat = np.concatenate([X,y], axis=1)

from sklearn.model_selection import train_test_split as tts
dat_train, dat_test = tts(dat, test_size=0.3, random_state=1234)


mae_lm = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  lm.fit(X_train.reshape(-1,1),y_train)
  yhat_lm = lm.predict(X_test.reshape(-1,1))
  mae_lm.append(mean_absolute_error(y_test, yhat_lm))
print("Validated MAE Linear Regression: ${:,.2f}".format(1000*np.mean(mae_lm)))


mae_lk = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lk = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.15)
  mae_lk.append(mean_absolute_error(y_test, yhat_lk))
print("Validated MAE Local Kernel Regression: ${:,.2f}".format(1000*np.mean(mae_lk)))


mae_xgb = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model_xgb.fit(X_train.reshape(-1,1),y_train)
  yhat_xgb = model_xgb.predict(X_test.reshape(-1,1))
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression: ${:,.2f}".format(1000*np.mean(mae_xgb)))
```
#### Final results: 

Validated MAE Linear Regression: $4,447.94      
Validated MAE Local Kernel Regression: $4,090.03      
Validated MAE XGBoost Regression: $4,179.17      


### Notes regarding the MSE and MAE: 
The goal of any machine learning model is to evaluate the accuracy of the model. In this project, the Mean Squared Error (MSE) and the Mean Absolute Error (MAE) are examined to evaluate the performance of the model in regression analysis.

The Mean Squared Error represents the average of the squared difference between the original and predicted values in the data set. It measures the variance of the residuals.

<img width="256" alt="image" src="https://user-images.githubusercontent.com/98488324/155912406-93bcb3f3-a79d-4363-9aaf-a2a57ee9fb71.png">
<img width="312" alt="image" src="https://user-images.githubusercontent.com/98488324/155914108-49ac10e0-6783-498e-b14a-9da52bf29af8.png">


The Mean Absolute Error represents the average of the absolute difference between the actual and predicted values in the dataset. It measures the average of the residuals in the dataset.

<img width="266" alt="image" src="https://user-images.githubusercontent.com/98488324/155912365-371ad65c-258a-40ed-8f91-93a086929533.png">



## References
Maklin, C. (May 9, 2020). [_Medium_](https://towardsdatascience.com/xgboost-python-example-42777d01001e). (https://towardsdatascience.com/xgboost-python-example-42777d01001e)

Chugh, A. (Dec 8, 2020). [_Medium_](https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e). (https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e)


##### Jekyll Themes
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/r-fukutoku/Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

##### Support or Contact
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
