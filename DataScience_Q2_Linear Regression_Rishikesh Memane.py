#!/usr/bin/env python
# coding: utf-8

# ### 2. Salary_hike -> Build a prediction model for Salary_hike

# In[10]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# #### Importing Dataset, EDA and Visulalisation.

# In[4]:


df = pd.read_csv('Salary_Data.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[11]:


sns.distplot(df['Salary'])
plt.title("Salary")


# In[13]:


sns.distplot(df['YearsExperience'])
plt.title("Years Experience")


# In[14]:


sns.pairplot(df)


# In[15]:


df.corr()


# #### As we can see from the pairplot, a linear realtionship can be observed between salary and experience. The correlation coefficient as well is also indicating high correlation between the same.
# 

# #### Building a Linear regression model

# In[16]:


model = smf.ols("Salary ~ YearsExperience",data=df).fit()
model.summary()


# In[17]:


sns.regplot(x= 'YearsExperience',y='Salary',data= df)


# ## Performing Log and sqrt trasformations to Check whether better rsquared values can be obtained.

# In[19]:


logmodel = np.log(df)
df.head()


# In[21]:


sqrtmodel= np.sqrt(df)
sqrtmodel.head()


# # Comparing Models before and after trasnforming

# In[58]:


from matplotlib.pyplot import figure
fig, ax =plt.subplots(1,3,figsize=(20,3))
sns.distplot(df['Salary'],ax=ax[0])
sns.distplot(logmodel['Salary'],ax=ax[1])
sns.distplot(sqrtmodel['Salary'],ax=ax[2])
ax[0].set_title("Original")
ax[1].set_title("Log Model")
ax[2].set_title("SQRT Model")
fig.show()


# In[59]:


from matplotlib.pyplot import figure
fig, ax =plt.subplots(1,3,figsize=(20,3))
sns.distplot(df['YearsExperience'],ax=ax[0])
sns.distplot(logmodel['YearsExperience'],ax=ax[1])
sns.distplot(sqrtmodel['YearsExperience'],ax=ax[2])
ax[0].set_title("Original")
ax[1].set_title("Log Model")
ax[2].set_title("SQRT Model")
fig.show()


# In[61]:


logmodel =smf.ols('YearsExperience ~ Salary',data = logmodel).fit()
logmodel.summary()


# In[62]:


sqrtmodel =smf.ols('YearsExperience ~ Salary',data = sqrtmodel).fit()
sqrtmodel.summary()


# In[67]:


table ={'Model':['Model','Log Model','Sqrt Model'],"Rsquared":[model.rsquared,sqrtmodel.rsquared,logmodel.rsquared]}
table1 = pd.DataFrame(table)
table1


# ## As we can see, the original model has a higher rsquared value at 0.95.Thus,  we will proceed with the same.

# #### Model Prediction

# In[70]:


new_data = pd.Series([5,8,10,14,15])
new_data


# In[71]:


data_pred = pd.DataFrame(new_data,columns = ["YearsExperience"])
data_pred


# In[72]:


model.predict(data_pred)

