#!/usr/bin/env python
# coding: utf-8

# # Task 1 : Prediction using Supervised ML
# 
# ## Problem Statement
# 
# ### We have to predict percentage of marks of a student based on the number of study hours.
# 
# ###  Author: Divya Joshi

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Getting Data

# In[19]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)


# # Inspecting and Analysing Data

# In[12]:


data.head(5)          #gives first 5 values in the dataset


# In[4]:


data.describe()     


# In[20]:


df.shape                #to check the shape of the dataset


# In[39]:


df.corr()              #to check correlation


# In[ ]:





# In[23]:


df.isnull().sum()     #to check null values


# # Visualising the data

# In[24]:


df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage', color='green', size=25)
plt.xlabel('Hours Studied', color='purple', size=15)
plt.ylabel('Percentage Score', color='purple',size=15 )


# # Training the Model

# In[26]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[31]:


from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# # Regression Line

# In[33]:


line = regressor.coef_*x+regressor.intercept_

plt.figure(figsize=(11, 7))
plt.scatter(x, y)
plt.plot(x, line)
plt.show()


# In[34]:


print(x_test)
y_pred = regressor.predict(x_test)


# In[35]:


dfAP = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    #Comparing Actual and Predicted Values
dfAP      


# In[38]:


from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('r2 Score Error', r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# # What will be the predicted score if a student studies for 9.25 hrs/day?

# In[37]:


req = regressor.predict([[9.25]])
req[0]


# In[ ]:




