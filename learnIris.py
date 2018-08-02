
# coding: utf-8

# # Iris dataset

# In[3]:


from IPython.display import IFrame
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', width=300, height=200)


# In[4]:


# import load_iris function from datasets module
from sklearn.datasets import load_iris


# In[5]:


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)


# In[6]:


# print the iris data
print(iris.data)


# In[7]:


# print the names of the four features
print(iris.feature_names)


# In[8]:


# print integers representing the species of each observation
print(iris.target)


# In[9]:


# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


# In[11]:


# check the types of the features and response
print(type(iris.data))
print(type(iris.target))


# In[12]:


# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)


# In[13]:


# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)


# In[14]:


# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# In[15]:


print(X.shape)
print(y.shape)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[18]:


print(knn)


# In[19]:


knn.fit(X, y)


# In[23]:


knn.predict([[3, 5, 4, 2]])


# In[24]:


X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)


# In[25]:


# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)


# In[26]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)

