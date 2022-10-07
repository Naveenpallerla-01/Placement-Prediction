#!/usr/bin/env python
# coding: utf-8

# # extracting data

# In[29]:


import numpy as np
import pandas as pd


# In[30]:


data=pd.read_csv('D:\placement prediction project\placementdata.csv')


# In[31]:


data.head(100)


# In[32]:


data.info()


# In[33]:


data.describe()


# # Data Splitting

# In[34]:


x=data.iloc[:,data.columns!='Placed']
y=data.iloc[:,data.columns=='Placed']


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2)


# In[36]:


x_train.head()


# In[37]:


x_train.info()


# # Decision Tree Classifier:
# 

# In[38]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(x_train,y_train)


# In[41]:


y_pred=DTC.predict(x_test)
y_pred


# In[57]:


model1=metrics.accuracy_score(y_test,y_pred)*100
print(model1)


# In[58]:


cnf_matrix=confusion_matrix(y_test,y_pred)
cnf_matrix


# # Random Forest Classifier :

# In[59]:


from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(n_estimators=150)
RFC.fit(x_train,y_train)


# In[60]:


y_pred=RFC.predict(x_test)
y_pred


# In[61]:


model2=metrics.accuracy_score(y_test,y_pred)*100
print(model2)


# # Support Vector Machine

# In[62]:


from sklearn import svm
support=svm.LinearSVC(random_state=43)
support.fit(x_train,y_train)


# In[63]:


y_pred=support.predict(x_test)
y_pred


# In[64]:


model3=metrics.accuracy_score(y_test,y_pred)*100
print(model3)


# # Naive Bayes

# In[65]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)


# In[66]:


y_pred=gnb.predict(x_test)
model4=metrics.accuracy_score(y_test,y_pred)*100
print(model4)


# # Comparison Graph

# In[68]:


import matplotlib.pyplot as plt
objects=('DT','RF','SVM','NB')
y_pos=np.arange(len(objects))
performance=[model1,model2,model3,model4]
plt.bar(y_pos,performance,align='center',alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Accuracy Rate')
plt.title('Comparison Graph')
plt.show()


# In[ ]:




