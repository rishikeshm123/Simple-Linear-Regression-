#!/usr/bin/env python
# coding: utf-8

# # Linear regression 

# #### 1. Delivery_time -> Predict delivery time using sorting time 

# In[8]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as py
import statsmodels.formula.api as smf


# #### Importing dataset, EDA and Visulisation

# In[3]:


df= pd.read_csv('delivery_time.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


sns.pairplot(df)


# ## Correlation

# In[7]:


df.corr()


# #### As we can see correlation coefficient is 0.82, thus we can say that the two variables are associated. Also,the pairplot indicates positive linear realtionship between Delivery time and sorting time.

# #### Building a Linear regression model

# In[10]:


df = df.rename({'Delivery Time':'Delivery_Time', 'Sorting Time':'Sorting_Time'},axis = 1)
df.head()


# In[12]:


model = smf.ols("Delivery_Time ~ Sorting_Time",data=df).fit()
model.summary()


# In[14]:


sns.regplot(x="Sorting_Time",y="Delivery_Time",data = df)


# ## Performing transformations to improve rsquared values.

# In[20]:


logdata = np.log(df)
logdata.head()


# In[22]:


df.head()


# In[23]:


logmodel=smf.ols('Delivery_Time	~Sorting_Time',data=logdata).fit()
logmodel.summary()


# In[24]:


sns.regplot(x="Sorting_Time",y="Delivery_Time",data = logdata)


# #### Model Prediction using log transformed model

# In[26]:


new_data= pd.Series([10,11, 12,15])
new_data


# In[27]:


data_pred = pd.DataFrame(new_data,columns= ["Sorting_Time"])
data_pred


# In[28]:


logmodel.predict(data_pred)


# In[ ]:




