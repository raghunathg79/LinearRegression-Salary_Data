#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.model_selection import train_test_split  


# In[3]:


SD=pd.read_csv('C:\\Users\\RAGHUNATH GANESAN\\Documents\\DATA SCIENCE\\ML_AI\\Nagaraj ML and AI\\Salary_Data.csv')


# In[4]:


SD.describe()


# In[5]:


SD.head()


# In[6]:


SD.tail()


# In[7]:


SD.isnull()


# In[8]:


plt.xlabel("YearsExperience", fontsize=24, color='red')
plt.ylabel("Salary", fontsize=24, color='green')
plt.scatter(SD['YearsExperience'],SD['Salary'])
fig= plt.figure(figsize=(18,6))


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


LinearReg=LinearRegression()


# In[11]:


X=SD[['YearsExperience']]
Y=SD[['Salary']]


# In[12]:


LinearReg.fit(X,Y)


# In[13]:


LinearReg.coef_


# In[14]:


X_test = np.arange(1,20, 1.5)


# In[15]:


X_test


# In[16]:


X_test=X_test.reshape(-1,1)


# In[17]:


X_test


# In[18]:


Y_pred=LinearReg.predict(X_test)


# In[19]:


Y_pred


# In[20]:


plt.xlabel("YearsExperience", fontsize=24, color='red')
plt.ylabel("Salary", fontsize=24, color='green')
plt.scatter(SD["YearsExperience"],SD["Salary"])
plt.scatter(X_test,Y_pred)
plt.plot(X_test,Y_pred)
fig= plt.figure(figsize=(14,7))


# # Polynominal Regression

# In[21]:


SD['New_Salary'] = SD['Salary'] + 2*SD['YearsExperience']*SD['YearsExperience']*2000


# In[22]:


SD.head()


# In[23]:


plt.xlabel("YearsExperience", fontsize=24, color='red')
plt.ylabel("Salary", fontsize=24, color='green')
plt.scatter(SD["YearsExperience"],SD["Salary"])
plt.scatter(SD["YearsExperience"],SD["New_Salary"])


# In[ ]:




