#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.kaggle.com/datasets/arshid/iris-flower-dataset


# ## Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the dataset

# In[16]:


columns=['sepal_length','sepal_width','petalplength','Petal_width','species']
#load the data
df=pd.read_csv('IRIS.csv')
df.head(150)


# ## Visualization of Dataset

# In[17]:


df.describe()


# In[18]:


#visualize the whole dataset
if 'species' not in df.columns:
    raise ValueError("Column 'species' does not exist in the dataframe")

# Create pairplot
sns.pairplot(df, hue='species')

# Show plot
plt.show()


# ## Separating input columns and the output columns

# In[19]:


#separate features and target 
data = df.values

X = data[:,0:4]
Y = data[:,4]


# ## Splitting the data into Training and Testing 

# In[25]:


# Split the data to train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
print(Y_test)


# ## Model1= vector machine algorithm

# In[26]:


# support vector machine algorithm
from sklearn.svm import SVC
model_svc= SVC()
model_svc.fit(X_train,Y_train)


# In[34]:


prediction1 = model_svc.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction1)*100)
for i in range(len(prediction1)):
    print(Y_test[i],prediction1[i])


# ## Model2: Logistic Regression

# In[35]:


# Logistic regression
from sklearn.linear_model import LogisticRegression
model_LR= LogisticRegression()
model_LR.fit(X_train,Y_train)


# In[36]:


prediction2=model_LR.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_test,prediction2)*100)


# ## Model3: Decision tree classifier

# In[40]:


#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train,Y_train)


# In[42]:


prediction3=model_svc.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_test,prediction3)*100)


# In[43]:


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction2))


# In[44]:


X_new = np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])
# prediction of the species from the input vector 
prediction = model_svc.predict(X_new)
print("prediction of Species: {}".format(prediction))


# In[ ]:




