#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv('ds_salaries.csv')


# In[31]:


data.head(6)


# In[30]:


data.columns


# In[7]:


dimension = data.shape
print(dimension)


# In[8]:


sns.heatmap(data.isna())


# In[11]:


correlation = data.corr()
print(correlation)


# In[14]:


sns.pairplot(data)


# In[20]:


plt.figure()
plt.subplot()
X = data['work_year']
Y = data['salary']
plt.scatter(X, Y)
plt.show()
plt.subplot()
X1 = data['company_size']
Y1 = data['salary']
plt.scatter(X1, Y1)
plt.show()


# In[23]:


df = data.copy()


# In[37]:


salary = df['salary']
salary.value_counts()


# In[38]:


job_title = df['job_title']
job_title.value_counts()


# In[43]:


data_1 = df.where(job_title == 'Data Scientist')
data_1.head()


# In[44]:


sns.heatmap(data_1.isna(), cbar = 'False')


# In[79]:


data_1.dropna(axis=0, inplace = True)
data_1.reset_index(drop = True, inplace = True)
data_1.head()


# In[66]:


data_1.shape


# In[48]:


sns.heatmap(data_1.isna(), cbar = 'False')


# In[80]:


salary_dataScientist = data_1['salary']
salary_dataScientist_in_usd = data_1['salary_in_usd']
salary_dataScientist.value_counts()


# In[50]:


salary_dataScientist.max()


# In[51]:


salary_dataScientist.min()


# In[52]:


salary_dataScientist.mean()


# In[145]:


plt.figure()
plt.subplots(figsize = (6,3))
plt.xlabel('experience_level')
plt.ylabel('salary')
X = data_1['experience_level']
Y = data_1['salary']
plt.scatter(X, Y)
plt.show()
plt.subplots(figsize = (6,3))
plt.xlabel('company_size')
plt.ylabel('salary')
X1 = data_1['company_size']
Y1 = data_1['salary']
plt.scatter(X1, Y1)
plt.show()


# In[72]:


sal_cur = data_1['salary_currency']
sal_cur.value_counts()


# In[91]:


salary_dataScientist_in_usd.plot.hist()


# In[114]:


salary_mean = data_1.groupby(['company_location']).mean()['salary']
salary_in_usd_mean = data_1.groupby(['company_location']).mean()['salary_in_usd']


# In[153]:


print(salary_mean)


# In[155]:


salary_mean.plot(figsize = (5,2.5), color = 'k')
plt.ylabel('salary_mean')
plt.figure()
salary_in_usd_mean.plot(figsize = (5,2.5), color = 'r')
plt.ylabel('salary_in_usd_mean')


# In[158]:


print(salary_in_usd_mean)


# In[159]:


salary_in_usd_mean.mean()


# In[138]:


location = data_1['company_location']
dt = data_1.where(location == 'CL')
dt.dropna(axis = 0, inplace = True)
dt.reset_index(drop = True, inplace = True)


# In[139]:


dt


# In[ ]:




