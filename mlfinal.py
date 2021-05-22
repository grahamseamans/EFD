# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:39:01 2021

@author: user
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import seaborn as sns

sns.set()

def fitplot_interpolation(model,x,y,name):
    """
    Function for fitting and plotting an interpolation model.
    
    Uses instantiated model, x, y, and a name for plot labeling.
    Returns a scatter plot of predictions and test data, as well as the fit
    model.
    """
    
    assert type(name) == str, "Name must be a string"
        
    assert len(x) == len(y), "x and y must be the same length"
        
    assert len(np.shape(x)) == 2, "x must be 2 dimensional"
    
    assert len(np.shape(y)) == 1, "y must be 1 dimensional"
        
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(x,y)
    
    model.fit(X_train_i,y_train_i)  
    preds = model.predict(X_test_i)
    
    plt.subplots()
    plt.title(name)
    plt.scatter(y_test_i,preds)
    plt.xlabel('True Wind Speed (m/s)')
    plt.ylabel('Predicted Wind Speed (m/s)')
    
    print("Mean Absolute Error " + name)
    print(mean_absolute_error(y_test_i,preds))
    
    print("Mean Squared Error " + name)
    print(mean_squared_error(y_test_i,preds))
    
    print("Explained Variance Score " + name)
    print(explained_variance_score(y_test_i, preds))
    
    return model
    
def fit_scat_ts_extrapolation(model,x,y,y_df,cutoff,name):
    """
    Function for fitting and plotting an extrapolation model.
    
    Uses instantiated model, x, y, y_df which yields timestamps, a cutoff value
    for separating training and testing data, and a name for plot labeling.
    The model is fit using the training data and then predicts the test data.
    The predictions are plotted against test data in a scatter plot. Also
    produces a time series showing test data and predictions. This function
    returns the fit model.
    """
    
    assert type(name) == str, "Name must be a string"
        
    assert type(cutoff) == int,"Cutoff must be an int"
        
    assert len(x) == len(y), "x and y must be the same length"
        
    assert len(np.shape(x)) == 2, "x must be 2 dimensional"
    
    assert len(np.shape(y)) == 1, "y must be 1 dimensional"
        
    assert cutoff >= 0, "Cutoff must be > 0"
        
    assert cutoff <= np.shape(x)[0], "Cutoff cannot be >= length of x"
    
    X_train = x[:cutoff,:]
    y_train = y[:cutoff]
    X_test = x[cutoff:,:]
    y_test = y[cutoff:]
    
    model.fit(X_train,y_train)  
    preds = model.predict(X_test)
    
    plt.subplots()
    plt.title(name)
    plt.scatter(y_test,preds)
    plt.xlabel('True Wind Speed (m/s)')
    plt.ylabel('Predicted Wind Speed (m/s)')
    
    plt.subplots()
    plt.title(name)
    plt.scatter(y_df.index[len(X_train):-1],y_test)
    plt.scatter(y_df.index[len(X_train):-1],preds)
    plt.ylabel('Wind Speed (m/s)')
    plt.legend(['y_test','y_pred'])
    
    print("Mean Absolute Error " + name)
    print(mean_absolute_error(y_test,preds))
    
    print("Mean Squared Error " + name)
    print(mean_squared_error(y_test,preds))
    
    print("Explained Variance Score " + name)
    print(explained_variance_score(y_test, preds))
    
    return model

lems = r'C:\Users\user\lemsdir' #Set to directory with LEMS data

os.chdir(lems) #Sets directory as directory with LEMS data

"""
Read in all lems data, place into lists of series based on type of measurement.
This extracts surface temperature, air temperature and wind speed from each 
sensor, then places it into a list which contains all sensors for that type of 
measurement. This section also extracts date info to create a date time index
for each series, allowing sensors to be synchronized
"""

ll = []
ll2 = []
yl = []
sens = []
for f in os.listdir(lems):
    lin = []
    lemsf = pd.read_csv(f)
    sens.append(f[:5])
    for i in lemsf.index:
        y = lemsf['Year'][i]
        m = lemsf['Month'][i]
        d = lemsf['Date'][i]
        h = lemsf['Hour'][i]
        mi = lemsf['Minute'][i]
        s = lemsf['Second'][i]
        ts = pd.Timestamp(year=y, month=m, day=d, hour=h, minute=mi, second=s)
        lin.append(ts)
    X = pd.Series(data=lemsf['MLX_IR_C'].values,index=lin) #Raw surface temperatures
    X2 = pd.Series(data=lemsf['MLX_Amb_C'].values,index=lin) #Raw Air temperature
    y = pd.Series(data=lemsf['Sonic_Spd'].values,index=lin) #Raw wind speed
    ll.append(X)
    ll2.append(X2)
    yl.append(y)



xdf = pd.concat(ll,axis=1) #Combine all surface measurements into DF
X = xdf.loc[(xdf.index >= '2019-07-01') #Raw surface temperature
                     & (xdf.index < '2019-07-10')].values #Select for days in early July


#Mean surface temperature across sensors
Xb = xdf.loc[(xdf.index >= '2019-07-01') & (xdf.index < '2019-07-10')].mean(axis=1).values[:-1]

Xd = np.roll(X,-1,axis=0)-X #Finite temperature difference in time at each sensor
Xd = Xd[:-1,:] #Last measurement must be removed since finite difference is 1 shorter

#Missing data is replaced by average difference across all sensors without missing data
for i in range(Xd.shape[0]):
    row = Xd[i,:]
    row[np.isnan(row)]=np.nanmean(row)
    Xd[i,:] = row

"""
#This allows for the mean temperature difference to be used, performs poorly for perceptron extrapolation
Xd = np.mean(Xd,axis=1)
"""

#Combine air temperatures into DF, select for times in early July
x2df = pd.concat(ll2,axis=1)
X2 = x2df.loc[(x2df.index >= '2019-07-01')
                     & (x2df.index < '2019-07-10')]

#Mean air temperature, last measurement removed to ensure same shape as ground temperatures
Xa = X2.mean(axis=1).values[:-1] 

#Select wind speed values only during July 1-10
ydf = pd.concat(yl,axis=1)
ydf = ydf.loc[(ydf.index >= '2019-07-01')
                     & (ydf.index < '2019-07-10')]

X = np.column_stack([Xd,Xa.reshape(-1,1),Xb]) #Combine all x values into array

ym = ydf.mean(axis=1).values #ym is the average wind speed at all sensors
ym = ym[:-1] #Remove final value to ensure same shape as finite differences

xyarr = np.column_stack([X,ym])
xydf = pd.DataFrame(index=ydf.index[:-1],
                    data=xyarr,columns = sens + ['Air Temperature','Surface Temperature','Wind Speed'])
"""
#To use mean surface temperature change
xydf = pd.DataFrame(index=ydf.index[:-1],
                    data=xyarr,columns = ['Delta T','Air Temperature','Surface Temperature','Wind Speed'])
"""
corr = xydf.corr()
f, ax = plt.subplots()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr)

#Extrapolation using perceptron
pipe_p_e = Pipeline([('scaler', StandardScaler()), 
                     ('neuralnet',MLPRegressor(random_state=1,max_iter=2000))])
pe = fit_scat_ts_extrapolation(pipe_p_e,X,ym,ydf,60000,'Perceptron Extrapolation Model')

#Extrpolation using linear regression
pipe_l_e = Pipeline([('scaler', StandardScaler()), ('linearregression', LinearRegression())])
le = fit_scat_ts_extrapolation(pipe_l_e,X,ym,ydf,60000,'Linear Extrapolation Model')


#Interpolation using perceptron
pipe_p_i = Pipeline([('scaler', StandardScaler()), 
                     ('neuralnet',MLPRegressor(random_state=1,max_iter=2000))])
pi = fitplot_interpolation(pipe_p_i,X,ym,'Perceprtron Interpolation Model')


#Interpolation using linear regression
pipe_l_i = Pipeline([('scaler', StandardScaler()), ('lineargregression',LinearRegression())])
li = fitplot_interpolation(pipe_l_i,X,ym,'Linear Interpolation Model')