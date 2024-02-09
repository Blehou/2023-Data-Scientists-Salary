import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

##### Chargement du dataset

data = pd.read_csv('ds_salaries.csv')

##### Phase de pre-processing

data.head(6)

Colonnes = data.columns # Afficher les colonnes du dataset
print(Colonnes)

dimension = data.shape
print('Dimension de data = ', dimension)

# Visualisation des valeurs manquantes
sns.heatmap(data.isna())

# Correlation entre les variables
correlation = data.corr()
print(correlation)

# Visualisation du dataset 
sns.pairplot(data)

##### Phase 
df = data.copy()
salary = df['salary']
salary.value_counts()

job_title = df['job_title']
job_title.value_counts()

# Recuperation des données uniquement pour les data scientists
data_1 = df.where(job_title == 'Data Scientist')

#Visualisation et suppression des valeurs manquantes
sns.heatmap(data_1.isna(), cbar = 'False')
data_1.dropna(axis=0, inplace = True)
data_1.reset_index(drop = True, inplace = True)
data_1.head()

sns.heatmap(data_1.isna(), cbar = 'False')

#Taille du nouveau dataset
dim = data_1.shape
print(dim)


# In[48]:





# In[80]:


salary_dataScientist = data_1['salary']
salary_dataScientist_in_usd = data_1['salary_in_usd']
salary_dataScientist.value_counts()




salary_dataScientist.max()

salary_dataScientist.min()

salary_dataScientist.mean()


#Figure montrant le salaire en fonction de la taille de l'entreprise et de l'année d'expérience
plt.figure(figsize = (8,4))
plt.subplot(1,2,1)

plt.xlabel('experience_level')
plt.ylabel('salary')
X = data_1['experience_level']
Y = data_1['salary']

plt.scatter(X, Y)
plt.show()

plt.subplot(1,2,2)
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




