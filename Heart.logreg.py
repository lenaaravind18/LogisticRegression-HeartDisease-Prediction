#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv("/Users/srinivasan/Downloads/heart (1).csv")


# In[6]:


data


# In[15]:


x=data.iloc[:,:-1]


# In[16]:


y=data.iloc[:,-1]


# In[19]:


print (data.shape)
print (x.shape)
print (y.shape)


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=5)


# In[23]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[24]:


predictions=model.predict(x_test)


# In[26]:


print("Accuracy",(model.score(x_test,y_test)))


# In[28]:


from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,predictions)
print(confusion_matrix)


# In[ ]:




