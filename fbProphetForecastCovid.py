#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings;
warnings.simplefilter('ignore')


# In[2]:


pip install pystan fbprophet


# In[3]:


import pandas as pd
import json
from fbprophet import Prophet
# importing the requests library 
import requests 
  
# api-endpoint 
URL = "https://coronavirusapi-france.now.sh/AllDataByDepartement?Departement=Bas-Rhin"
  
# location given here 
location = "delhi technological university"
  
# defining a params dict for the parameters to be sent to the API 
PARAMS = {'address':location} 
  
# sending get request and saving the response as response object 
r = requests.get(url = URL) 
  
# extracting data in json format 
data = r.json() 
  
df = pd.json_normalize(data['allDataByDepartement'])
basrhin = df
csvd= basrhin.to_csv('new.csv',index=False)

df = pd.read_csv('new.csv')


# In[4]:


df = pd.read_csv('new.csv')


# In[5]:


df.head()


# In[6]:


df1 = df[['date', 'nouvellesHospitalisations']]
df1.columns = ['ds', 'y']
df1.head()


# In[7]:


m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
model = m.fit(df1)


# In[8]:


future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
forecast.head()


# In[9]:


forecast.tail()


# In[10]:


plot1 = m.plot(forecast)


# In[11]:


plot2 = m.plot_components(forecast)


# In[ ]:




