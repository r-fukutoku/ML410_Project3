Create a new Github page with a presentation on the concepts of Multivariate Regression Analysis and Gradient Boosting. Include a presentation of Extreme Gradient Boosting (xgboost).
Apply the regression methods (including lowess and boosted lowess) to real data sets, such as "Cars" and "Boston Housing Data".  Record the cross-validated mean square errors and the mean absolute errors.

For each method and data set report the crossvalidated mean square error and 
determine which method is achieveng the better results.
In this paper you should also include theoretical considerations, examples of Python coding and plots. 
The final results should be clearly stated.


# Concepts and Applications of Multivariate Regression Analysis and Gradient Boosting (Extreme Gradient Boosting (xgboost))

### Locally Weighted Regression (Abbreviated "loess" ; "lowess")
Locally Weighted Regression is a non-parametric regression method that combines multiple regression models in a meta-model based on k-nearest neighbor. It addresses situations in which the classical procedures do not perform well or cannot be effectively applied without undue labor. 

Lowess combines much of the simplicity of linear least squares regression with the flexibility of nonlinear regression. 
It does this by fitting simple models to localized subsets of the data to build up a function that describes the variation in the data, point by point.

With an equation, the main idea of linear regression can be expressed as the assumption that:

<img width="217" alt="image" src="https://user-images.githubusercontent.com/98488324/153696127-71453565-f03a-4b04-bddf-9512e2e60332.png">

If we pre-multiply this equation with a matrix of weights (the "weights" are on the main diagonal and the rest of the elements are 0), we get:

<img width="349" alt="image" src="https://user-images.githubusercontent.com/98488324/153696139-ea560e34-528f-4683-a81d-94f40ba5276b.png">

The distance between two independent observations is the Euclidean distance between the two represented  𝑝− dimensional vectors. The equation is:

<img width="651" alt="image" src="https://user-images.githubusercontent.com/98488324/153696145-f262158e-dd10-47d4-a9b8-cc5b133fb807.png">

We shall have  𝑛  differrent weight vectors because we have  𝑛  different observations.

__Important aspect__: linear regression can be seen as a linear combination of the observed outputs (values of the dependent variable).

We have:

<img width="276" alt="image" src="https://user-images.githubusercontent.com/98488324/153696158-eb25a77b-e093-41fb-8c17-3463d390becf.png">

We solve for  𝛽  (by assuming that  𝑋𝑇𝑋  is invertible):

<img width="428" alt="image" src="https://user-images.githubusercontent.com/98488324/153696164-f8c5920b-06b9-4792-8955-56e82a22e692.png">

We take the expected value of this equation and obtain:

<img width="231" alt="image" src="https://user-images.githubusercontent.com/98488324/153696168-10bfee30-6312-4f6e-a173-75625e60b8cf.png">

Therefore, the predictions we make are:

<img width="249" alt="image" src="https://user-images.githubusercontent.com/98488324/153696171-65bf808d-641b-4bf3-9269-fe3a4694c897.png">

For the locally weighted regression, we have:

<img width="316" alt="image" src="https://user-images.githubusercontent.com/98488324/153696198-b83fc443-af89-444b-9a19-cfa95d154109.png">

__The big idea__: the predictions we make are a linear combination of the actual observed values of the dependent valuable.

For locally weighted regression,  𝑦̂   is obtained as a different linear combination of the values of y. 


### Random Forest
Random Forest Regression is a supervised learning algorithm that utilizes ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.

![image](https://user-images.githubusercontent.com/98488324/153693726-36f3fe10-9648-4606-92cb-293b6c78a9dd.png)

This diagram is the basic structure of a Random Forest. The trees are parallel with no interaction amongst them. A Random Forest operates by constructing several decision trees during training time and outputting the mean of the classes as the prediction of all the trees. 

By default, the decision trees we use here will make their predictions based on the mean value of the target within each leaf of the tree, and the splitting criteria will be based on minimizing the mean square error, MSE.



### Applications with Real Data
Cars Data (output varable (y) is the mileage (MPG)): 

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
cars = pd.read_csv("drive/MyDrive/DATA410_AdvML/cars.csv")
```
<img width="234" alt="image" src="https://user-images.githubusercontent.com/98488324/153694695-0e275da1-6379-44db-af1b-92a43d0c0544.png">


#### Locally Weighted Regression:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120

# import libraries
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)

# Locally Weighted Regression
def lowess_reg(x, y, xnew, kern, tau):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # IMPORTANT: we expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
    
x = cars['WGT'].values
y = cars['MPG'].values

lowess_reg(x,y,2100,tricubic,0.01)

xnew = np.arange(1500,5500,10) 
yhat = lowess_reg(x,y,xnew,tricubic,80)

plt.scatter(x,y)
plt.plot(xnew,yhat,color='red',lw=2)
```

<img width="644" alt="image" src="https://user-images.githubusercontent.com/98488324/153696320-25bb092f-ef78-4f8b-b819-41dd99550893.png">


```python
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)

scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))

yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scaled,tricubic,0.1)

print(mse(yhat_test,ytest))
```
15.961885966790936


```python
plt.plot(np.sort(xtest_scaled.ravel()),yhat_test)
```
![image](https://user-images.githubusercontent.com/98488324/153695299-b5a1f418-3757-4854-ab90-0a4184959d79.png)


#### Random Forest:
```python
rf = RandomForestRegressor(n_estimators=100,max_depth=3)
rf.fit(xtrain_scaled,ytrain)

print(mse(ytest,rf.predict(xtest_scale)))
```
15.931305250431844

```python
yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scale.ravel(),tricubic,0.1)

dat_test = np.column_stack([xtest_scale,ytest,yhat_test])

sorted_dat_test = dat_test[np.argsort(dat_test[:,0])]

kf = KFold(n_splits=10,shuffle=True,random_state=310)
mse_lwr = []
mse_rf = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_lwr = lowess_reg(xtrain.ravel(),ytrain,xtest.ravel(),tricubic,0.5)
  rf.fit(xtrain,ytrain)
  yhat_rf = rf.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_rf.append(mse(ytest,yhat_rf))
```

#### Final results: 

```python
print("Crossvalidated mean square error of Lowess is " + str(np.mean(mse_lwr)))
```
Crossvalidated mean square error of Lowess is 17.584499477691253

```python
print("Crossvalidated mean square error of Random Forest is " + str(np.mean(mse_rf)))
```
Crossvalidated mean square error of Random Forest is 18.3197148440588

Since we aim to minimize the crossvalidated mean square error (MSE) for the better results, I conclude that Locally Weighted Regression (Lowess) achieved the better result than Random Forest. 



## References
Bakshi, C. (Jun 8, 2020). [_Medium_](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84). (https://levelup.gitconnected.com/random-forest-regression-209c0f354c84)


Sicotte, X. (May 24, 2018).
[_Data Blog_](https://xavierbourretsicotte.github.io/loess.html). (https://xavierbourretsicotte.github.io/loess.html)


##### Jekyll Themes
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/r-fukutoku/Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

##### Support or Contact
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
