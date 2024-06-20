#!/usr/bin/env python
# coding: utf-8

# In[176]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[177]:


df_train = pd.read_csv(r'C:\Users\MANISH\Desktop\BlackFriday_FE_EDA\train.csv')


# In[178]:


df_train.head()


# In[179]:


df_train.info()


# In[180]:


df_train.describe()


# In[181]:


df_train.isnull().sum()


# In[182]:


#Importing TEST DATA

df_test = pd.read_csv(r'C:\Users\MANISH\Desktop\BlackFriday_FE_EDA\test.csv')


# In[183]:


df_test.head()


# In[184]:


df_train.columns


# In[185]:


df_test.columns


# In[186]:


frame = [df_train , df_test]
df = pd.concat(frame)


# In[187]:


df


# In[188]:


df.info()


# In[189]:


df.describe()


# In[190]:


df.drop(['User_ID'] , axis =1 , inplace = True)


# In[191]:


df.head()


# In[192]:


#Handling Categorial Feature  : Gender
df['Gender'] = df['Gender'].map({'F': 0 , 'M' : 1})
df.head()


# In[193]:


#Handling Categorial Feature  : Age
df['Age'].unique()


# In[194]:


df['Age'] = df['Age'].map({'0-17':1 ,'18-25':2 ,'26-35':3, '36-45':4 , '46-50':5 , '51-55':6 , '55+':7 })
df.head()


# In[195]:


#Fixingg categorial city_category
pd.get_dummies(df['City_Category'])


# In[196]:


df_city=pd.get_dummies(df['City_Category'] , drop_first = True)


# In[197]:


df_city.head()


# In[198]:


df = pd.concat([df,df_city] , axis =1 )


# In[199]:


df.head()


# In[200]:


df.drop('City_Category' , axis =1, inplace = True )


# In[201]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows  
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['B']= label_encoder.fit_transform(df['B']) 
  
df['B'].unique()


# In[202]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows  
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['C']= label_encoder.fit_transform(df['C']) 
  
df['C'].unique()


# In[203]:


df.head()


# In[204]:


#Missing Values
df.isnull().sum()


# In[205]:


df['Product_Category_2'].unique()


# In[206]:


df['Product_Category_2'].value_counts()


# In[207]:


df['Product_Category_2'].mode()[0]


# In[208]:


df['Product_Category_2'] = df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[209]:


df['Product_Category_2'].isnull().sum()


# In[210]:


#Product Category 3 Replace missing values
df['Product_Category_3'].unique()


# In[211]:


df['Product_Category_3'].value_counts()


# In[212]:


df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[213]:


df['Product_Category_3'].isnull().sum()


# In[214]:


df.shape


# In[215]:


df.head()


# In[216]:


df['Stay_In_Current_City_Years'].unique()


# In[217]:


df['Stay_In_Current_City_Years']= df['Stay_In_Current_City_Years'].str.replace('+' , '')


# In[218]:


df.head()


# In[219]:


df.info()


# In[220]:


#Convert Object into Integers
df['Stay_In_Current_City_Years']= df['Stay_In_Current_City_Years'].astype(int)


# In[221]:


df.info()


# In[222]:


sns.barplot(x='Age' , y='Purchase' , hue='Gender' , data=df)


# In[223]:


##Visualization of purchase with occupation

sns.barplot(x='Occupation' , y='Purchase' , hue='Gender' , data=df)


# In[224]:


##Visualization of purchase with Product_category_1
sns.barplot(x='Product_Category_1' , y='Purchase' , hue='Gender' , data=df)


# In[225]:


##Visualization of purchase with Product_category_2
sns.barplot(x='Product_Category_2' , y='Purchase' , hue='Gender' , data=df)


# In[226]:


##Visualization of purchase with Product_category_3
sns.barplot(x='Product_Category_3' , y='Purchase' , hue='Gender' , data=df)


# # FEATURE SCALING

# In[227]:


df_test = df[df['Purchase'].isnull()]


# In[228]:


df_test


# In[229]:


df_train = df[~ df['Purchase'].isnull()]


# In[230]:


df_train


# In[231]:


X =df_train.drop('Purchase' , axis =1)


# In[232]:


X.head()


# In[233]:


y = df_train['Purchase']


# In[234]:


y.head()


# In[235]:


X.shape


# In[236]:


y.shape


# In[237]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=104, test_size=0.25)


# In[239]:


X_train.drop('Product_ID' , axis =1 , inplace = True)
X_test.drop('Product_ID' , axis =1 , inplace = True)


# In[241]:


#Feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[242]:


#Train Model ----> LINEAR REGRESSION
from sklearn import linear_model, metrics
model = linear_model.LinearRegression() 
model.fit(X_train, y_train)


# In[243]:


# regression coefficients 
print('Coefficients: ', model.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(model.score(X_test, y_test)))


# In[244]:


# plot for residual error 
  
# setting plot style 
plt.style.use('fivethirtyeight') 
  
# plotting residual errors in training data 
plt.scatter(model.predict(X_train), 
            model.predict(X_train) - y_train, 
            color="green", s=10, 
            label='Train data') 
  
# plotting residual errors in test data 
plt.scatter(model.predict(X_test), 
            model.predict(X_test) - y_test, 
            color="blue", s=10, 
            label='Test data') 
  
# plotting line for zero residual error 
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2) 
  
# plotting legend 
plt.legend(loc='upper right') 
  
# plot title 
plt.title("Residual errors") 
  
# method call for showing the plot 
plt.show() 


# In[245]:


print(model.score(X_test, y_test))


# In[ ]:




