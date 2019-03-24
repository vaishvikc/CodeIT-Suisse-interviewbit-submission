
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


# Uploading the data
train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv")
train.head()


# In[12]:


# Makeing the train and test set
y_train = train['bestSoldierPerc']
x_train = train.drop(['bestSoldierPerc','soldierId','shipId','attackId'],1)

x_test = test.drop(['soldierId','shipId','attackId','Unnamed: 0','index'],1)

print (x_train.shape , x_test.shape)


# In[13]:


# Model

reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)


# In[20]:


y_test_pred = reg.predict(x_test)
print(y_test_pred)


# In[22]:


#Solution


submission = pd.DataFrame({'soldierId':test['soldierId'], 'bestSoldierPerc':y_test_pred})
submission = submission[['soldierId', 'bestSoldierPerc']]
submission.head()


submission.to_csv("sub_try_1.csv", index=False)

