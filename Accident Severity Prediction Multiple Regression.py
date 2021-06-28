#!/usr/bin/env python
# coding: utf-8

# In[225]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import seaborn as sns
import datetime as dt
import seaborn as sns
from statsmodels.formula.api import logit
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statistics import mean
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf


# In[226]:


data = pd.read_csv('US-Accidents.csv')
df = data
df['Weather_Condition'] = df['Weather_Condition'].replace(['Fair','Clear','Fair / Windy'],int(1))
df['Weather_Condition'] = df['Weather_Condition'].replace(['Mostly Cloudy', 'Partly Cloudy', 'Cloudy', 'Overcast', 
                                                           'Scattered Clouds', 'Light Rain', 'Haze', 'Fog', 'Cloudy / Windy'
                                                           ,'Light Drizzle', 'Smoke', 'Mist', 'Light Freezing Rain', 
                                                           'Partly Cloudy / Windy', 'Thunder in the Vicinity', 
                                                           'Thunder', 'Drizzle', 'Light Rain with Thunder', 
                                                           'Patches of Fog', 'Shallow Fog', 'Mostly Cloudy / Windy', 
                                                           'Light Rain Shower', 'Haze / Windy', 'Drizzle and Fog'],int(2))
df['Weather_Condition'] = df['Weather_Condition'].replace(['Light Snow', 'Rain', 'Snow', 'Light Thunderstorms and Rain', 
                                                           'Light Snow / Windy', 'Rain / Windy'],int(3))
df['Weather_Condition'] = df['Weather_Condition'].replace(['Heavy Rain', 'Thunderstorms and Rain', 'T-Storm', 
                                                           'Wintry Mix', 'Heavy Snow', 'Heavy T-Storm', 'Thunderstorm', 
                                                           'Heavy Thunderstorms and Rain'],int(4))
df = df.drop(columns=['ID', 'End_Time','End_Lng','End_Lat','Distance(mi)','Description','Number','Street','Country','Timezone',
                 'Airport_Code','Weather_Timestamp','Wind_Chill(F)','Wind_Direction','Precipitation(in)','Amenity',
                 'Bump','Crossing','Give_Way','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming',
                 'Turning_Loop','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'])
df['Side'] = df['Side'].replace('R',1)
df['Side'] = df['Side'].replace('L',0)
df['Junction'] = df['Junction'].replace(False,0)
df['Junction'] = df['Junction'].replace(True,1)
df['Traffic_Signal'] = df['Traffic_Signal'].replace(False,0)
df['Traffic_Signal'] = df['Traffic_Signal'].replace(True,1)
df['Sunrise_Sunset'] = df['Sunrise_Sunset'].replace('Night',0)
df['Sunrise_Sunset'] = df['Sunrise_Sunset'].replace('Day',1)
df = df.drop(columns=['Start_Time'])
df = df.dropna(axis=0)
df = df.drop(columns = ['State','Start_Lat','Start_Lng','City','County','Zipcode'])
df = df.rename(columns={"Temperature(F)": "Temperature", "Humidity(%)": "Humidity", "Pressure(in)": "Pressure", "Visibility(mi)" : "Visibility", "Wind_Speed(mph)" : "Wind_Speed"}, errors="raise")
df


# In[227]:


df.dtypes


# In[228]:


'''from collections import Counter
x = df['Traffic_Signal']
#z=np.exp(x)/(1+np.exp(x)) # Logistic function transforms x to z
y = df['Severity']

c = Counter(zip(x,y))
# create a list of the sizes, here multiplied by 10 for scale
s = [.3*c[(xx,yy)] for xx,yy in zip(x,y)]

# plot it
plt.grid(True)
plt.scatter(x, y, s=s)'''


# In[229]:


'''data = df
data['Severity'] = data['Severity'].replace(1,0)
data['Severity'] = data['Severity'].replace(2,0.333)
data['Severity'] = data['Severity'].replace(3,0.666)
data['Severity'] = data['Severity'].replace(4,1)
x = data['Traffic_Signal']
#z=np.exp(x)/(1+np.exp(x)) # Logistic function transforms x to z
y = data['Severity']

c = Counter(zip(x,y))
# create a list of the sizes, here multiplied by 10 for scale
s = [.3*c[(xx,yy)] for xx,yy in zip(x,y)]

# plot it
plt.grid(True)
plt.scatter(x, y, s=s)'''


# In[230]:


df1 = df
df1['Weather_Condition'] = df1['Weather_Condition'].astype(str)
df1['Side'] = df1['Side'].astype(str)
df1['Sunrise_Sunset'] = df1['Sunrise_Sunset'].astype(str)
df1['Junction'] = df1['Junction'].astype(str)
df1['Traffic_Signal'] = df1['Traffic_Signal'].astype(str)
df['Side'] = df['Side'].astype(int)
df.dtypes


# In[231]:


df1['Weather_Condition'].values


# In[232]:


#X = df[['Side', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
       #'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition', 'Junction',
       #'Traffic_Signal', 'Sunrise_Sunset']]

#X = sm.add_constant(X)
model = ols(formula = 'Severity ~ 1 + Side + Temperature + Humidity + Pressure + Visibility + Wind_Speed + Weather_Condition + Junction + Traffic_Signal + Sunrise_Sunset', data = df1).fit()

#model = sm.OLS(df['Severity'], X).fit()
yhat = model.predict(df1.iloc[:,1:11])
model.summary()


# In[210]:


print(model.summary())


# In[211]:


model.pvalues


# In[236]:


fig, ax = plt.subplots(figsize=(10,10))
df1['predicted'] = model.predict(df1.iloc[:,1:11])
plt.plot(model.predict(df1.iloc[:,1:11]),'o',alpha=0.15,label='predicted',markersize=10)
df1 = df1.sort_values(by=['predicted']) 
df1 = df1.reset_index()
df1 = df1.drop(columns = ['index'])
plt.plot(df1['Severity'],'o',alpha=0.1,color='red',label='actual')
plt.ylabel('Severity')
plt.legend()
#sort severity and plot also sort prediction


# In[237]:


fig, ax = plt.subplots(figsize=(10,10))
df1['predicted'] = model.predict(df1.iloc[:,1:11])
plt.plot(model.predict(df1.iloc[:,1:11]),'o',alpha=0.15,label='predicted',markersize=10)
df1 = df1.sort_values(by=['Severity']) 
df1 = df1.reset_index()
df1 = df1.drop(columns = ['index'])
plt.plot(df1['Severity'],'o',alpha=0.1,color='red',label='actual')
plt.ylabel('Severity')
plt.grid(True)
plt.title('Sorted Model Prediction Density Bubble Scatter Plot')
plt.legend()


# In[217]:


from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10,10))
df1['predicted'] = model.predict(df1.iloc[:,1:11])
#plt.plot(model.predict(df1.iloc[:,1:11]),'o',alpha=0.15,label='predicted',markersize=10)
df1 = df1.sort_values(by=['predicted']) 
df1 = df1.reset_index()
df1 = df1.drop(columns = ['index'])

# Generate fake data
x = range(0,2660)
y = df1['Severity']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x,df1['predicted'], c=z, s=10)
plt.show()


# In[150]:


plt.boxplot(df1['predicted'])


# In[26]:


df1.boxplot(column='predicted',by='Severity',figsize=(10,10))


# In[ ]:


df


# In[ ]:


df['Severity']


# In[ ]:


from collections import Counter
#x = df['Traffic_Signal']
#z=np.exp(x)/(1+np.exp(x)) # Logistic function transforms x to z
x = []
for i in range (0,2660):
    x.append(i)
y = df['Severity']

c = Counter(zip(x,y))
# create a list of the sizes, here multiplied by 10 for scale
s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]

# plot it
plt.grid(True)
plt.scatter(x, y, s=s)


# In[ ]:


plt.plot(df['Severity'],'o',alpha=0.2,color='red')


# In[44]:


# Forward Selection
X = df[['Side', 'Temperature', 'Humidity', 'Pressure',
       'Visibility', 'Wind_Speed', 'Weather_Condition', 'Junction',
       'Traffic_Signal', 'Sunrise_Sunset']]
mse = []
predictors = list(X.columns[:])

for i in range(len(predictors)):
    x = sm.add_constant(X[predictors[i]])
    model = sm.OLS(df['Severity'], x.astype(float)).fit()
    yhat = model.predict(x)
    mse.append(mean_squared_error(df['Severity'], yhat))

plt.plot(predictors, mse)
plt.xticks(rotation=45)
best = predictors[mse.index(min(mse))]
min(mse)
plt.title('One Predictor')
plt.ylabel('Mean-Squared Error')


# In[45]:


# Top 2
mse2 = []
for i in range(len(predictors)):
    if i is 8:
        mse2.append(.3)
    else:
        x2 = sm.add_constant(X[[best, predictors[i]]])
        model = sm.OLS(df['Severity'], x2.astype(float)).fit()
        yhat = model.predict(x2)
        mse2.append(mean_squared_error(df['Severity'], yhat))

plt.plot(predictors, mse2)
plt.xticks(rotation=45)
best2 = predictors[mse2.index(min(mse2))]
best2
plt.title('Two Predictors')
plt.ylabel('Mean-Squared Error')


# In[46]:


mse3 = []
for i in range(len(predictors)):
    if i is 0 or i is 8:
        mse3.append(.3)
    else:
        x3 = sm.add_constant(X[[best, best2, predictors[i]]])
        model = sm.OLS(df['Severity'], x3.astype(float)).fit()
        yhat = model.predict(x3)
        mse3.append(mean_squared_error(df['Severity'], yhat))

plt.plot(predictors, mse3)
plt.xticks(rotation=45)
best3 = predictors[mse3.index(min(mse3))]
best3
plt.title('Three Predictors')
plt.ylabel('Mean-Squared Error')


# In[114]:


# Setup/Randomize total dataset
import operator as op
df1 = df.sample(frac=1).reset_index(drop=True)
k = 5
sub = int(df1.shape[0] / k)      # Subset = total_rows / k_folds
mse1 = []
mse2 = []
mse3 = []
mse4 = []
mse5 = []

for i in range(0, 5):
    a, b = i*sub, (i+1)*sub
    test = df1.iloc[a:b]
    train = df1.drop(range(a, b))

    # Severity ~ 1 + Traffic_Signal
    X = sm.add_constant(train['Traffic_Signal'])
    Y = sm.add_constant(test['Traffic_Signal'])
    model1 = sm.OLS(train['Severity'], X).fit()
    yhat1 = model1.predict(Y)
    mse1.append(mean_squared_error(test['Severity'], yhat1))

    # Severity ~ 1 + Traffic_Signal + Side
    X = sm.add_constant(train[['Traffic_Signal', 'Side']])
    Y = sm.add_constant(test[['Traffic_Signal', 'Side']])
    model2 = sm.OLS(train['Severity'], X).fit()
    yhat2 = model2.predict(Y)
    mse2.append(mean_squared_error(test['Severity'], yhat2))

    # Severity ~ 1 + Traffic_Signal + Side + Weather_Condition
    X = sm.add_constant(train[['Traffic_Signal', 'Side', 'Weather_Condition']])
    Y = sm.add_constant(test[['Traffic_Signal', 'Side', 'Weather_Condition']])
    model3 = sm.OLS(train['Severity'], X).fit()
    yhat3 = model3.predict(Y)
    mse3.append(mean_squared_error(test['Severity'], yhat3))

    # All Predictors
    Xm = sm.add_constant(train.iloc[:, 1:11])
    Ym = sm.add_constant(test.iloc[:, 1:11])
    model4 = sm.OLS(train['Severity'], Xm).fit()
    yhat4 = model4.predict(Ym)
    mse4.append(mean_squared_error(test['Severity'], yhat4))

    # Only Intercept
    model5 = sm.OLS(train['Severity'], Xm['const']).fit()
    yhat5 = model5.predict(Ym['const'])
    mse5.append(mean_squared_error(test['Severity'], yhat5))
mean_test_mse = {'Pred: Traffic_Signal': mean(mse1), 
                 'Pred: Traffic_Signal, Side': mean(mse2),
                 'Pred: Traffic_Signal, Side, Weather_Condition': mean(mse3),
                 'All Predictors': mean(mse4),
                 'Only Intercept': mean(mse5)}
meanmse = [mean(mse1),mean(mse2),mean(mse3),mean(mse4),mean(mse5)]
nnn = ['Traffic_Signal','Traffic_Signal + Side',
       'Traffic_Signal + Side + Weather_Condition','All Predictors','Only Intercept']
errors = [np.std(mse1),np.std(mse2),np.std(mse3),np.std(mse4),np.std(mse5)]
std =  {'Pred: Traffic_Signal': np.std(mse1), 
                 'Pred: Traffic_Signal, Side': np.std(mse2),
                 'Pred: Traffic_Signal, Side, Weather_Condition': np.std(mse3),
                 'All Predictors': np.std(mse4),
                 'Only Intercept': np.std(mse5)}
std


# In[115]:


mean_test_mse


# In[69]:


plt.grid(True)
plt.bar(nnn,meanmse)
plt.xticks(rotation=90)


# In[76]:


#%matplotlib inline
fig, ax = plt.subplots(figsize=(10,10))
ax.bar(nnn, meanmse,
       yerr=errors,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)

ax.set_xticks(nnn)
ax.set_xticklabels(nnn)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=75)
plt.gca().set_ylim(bottom=.25)
plt.gca().set_ylim(top=.3415)
ax.set_title('Mean MSE per Model')
ax.set_ylabel('Mean Squared Error (5 folds)')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()


# In[79]:


fig, ax = plt.subplots(figsize=(10,10))
ax.bar(nnn, errors,
       align='center',
       color='red',
       alpha=0.5,
       ecolor='black',
       capsize=10)

ax.set_xticks(nnn)
ax.set_xticklabels(nnn)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=75)
plt.gca().set_ylim(bottom=.0175)
ax.set_title('MSE STD')
ax.set_ylabel('STD (5 folds)')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()


# In[80]:


fig, ax = plt.subplots(figsize=(10,10))
ax.bar(nnn, meanmse,
       yerr=errors,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)

ax.set_xticks(nnn)
ax.set_xticklabels(nnn)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=75)
#plt.gca().set_ylim(bottom=.273)
ax.set_title('Mean MSE per Model')
ax.set_ylabel('Mean Squared Error (5 folds)')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()


# In[ ]:


train.iloc[:, 1:11]


# In[107]:


Xm = sm.add_constant(df[['Traffic_Signal', 'Side', 'Weather_Condition']])
Ym = sm.add_constant(df[['Traffic_Signal', 'Side', 'Weather_Condition']])
model4 = sm.OLS(df['Severity'], Xm).fit()
yhat4 = model4.predict(Ym)
mean_squared_error(df['Severity'], yhat4)


# In[ ]:




