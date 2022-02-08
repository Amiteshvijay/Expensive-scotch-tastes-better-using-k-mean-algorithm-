#!/usr/bin/env python
# coding: utf-8

# # Apply K-means clustering in the attached scotch review data. Try to answer "Expensive scotch tastes better?"

# In[1]:


#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#read the data using pandas
df=pd.read_csv("scotch_review.csv")


# In[3]:


#read the top five data in datagrame
df.head()


# In[4]:


#display the last five data in dataframe
df.tail()


# In[5]:


#Display "review.point" and "price" in scatter plot 
plt.scatter(df['review.point'],df['price'],color='red')
plt.xlabel('review.point')
plt.ylabel('price')


# In[6]:


#We drop two columns "Unnamed" and "currency".This doesnot effect on your data.
df=df.drop(['Unnamed: 0','currency'],inplace=False, axis=1)


# In[7]:


#we drop the rows 19
df=df.drop(19)


# In[8]:


df.dtypes


# In[9]:


#print shape of dataset with rows and columns
print(df.shape)


# In[10]:


df.head()


# In[11]:


#copy the original data
df1=df.copy()


# In[12]:


# change the "review.point" data point to int  from object.
df1[['review.point']] = df1[['review.point']].astype(int)


# In[13]:


df1['price']=pd.Series(df1['price'])
df1['price']=pd.to_numeric(df1['price'], errors='coerce')
#print(pd.to_numeric(df1['price'], errors='coerce'))


# In[14]:


#Display the data types of data in dataframe
df1.dtypes


# In[15]:


#information of data
df1.info()


# In[16]:


df1.head()


# In[17]:


df1.tail()


# In[18]:


#df1['price']=df1['price'].fillna(0)
#df1['price']=pd.to_numeric(df1['price'], errors='coerce')
df1['price'] = df1['price'].fillna(df1['price'].median())


# In[ ]:





# In[19]:


df1.iloc[:,[2,3]].isnull().sum()


# In[20]:


X=df1.iloc[:,[2,3]].values


# In[21]:


#Using the elbow method to find the optimal number of clusters
df
from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[22]:


#Fitting K-MEans to the dataset
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)


# In[23]:


#Visualize the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster1')
#plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster2')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('cluster of scotch')
plt.xlabel('review.point')
plt.ylabel('price')
plt.legend()
plt.show()


# # Preprocessing using min max scaler

# In[24]:


scaler = MinMaxScaler()

scaler.fit(df1[['price']])
df1['price'] = scaler.transform(df1[['price']])

scaler.fit(df1[['review.point']])
df1['review.point'] = scaler.transform(df1[['review.point']])


# In[25]:


df1.head()


# In[26]:


kmeans.cluster_centers_


# # Apply k means clustering Algorithm

# In[27]:


X=df1.iloc[:,[2,3]].values


# In[28]:


#Using the elbow method to find the optimal number of clusters
df
from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[29]:


plt.scatter(df1['review.point'],df1['price'])
plt.xlabel('review.point')
plt.ylabel('price')


# In[30]:


#Fitting K-MEans to the dataset
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)


# In[31]:


kmeans = KMeans(n_clusters=3)
y_predicted = kmeans.fit_predict(X)
y_predicted


# In[32]:


#Visualize the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('cluster of scotch')
plt.xlabel('review.point')
plt.ylabel('price')
plt.legend()
plt.show()


#  from the above diagram 
#     
#     **It is clearly visible that "Expensive scotch tastes better" 

# In[ ]:




