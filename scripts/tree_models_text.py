#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score # will be used to separate training and test


# # Classifiers introduction - Exercise text
# 
# In the following program we introduce the basic steps of classification of a dataset in a matrix

# Importing the packages for this exercise.

# Define the matrix containing the data (one example per row) and the vector containing the corresponding target value.

# In[2]:


# simple initial dataset
X = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
Y = [1, 0, 0, 0, 1, 1]


# Declare the classification model you want to use and then fit the model to the data.

# In[3]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


# Predict the target value (and print it) for the passed data, using the fitted model currently in clf.

# In[4]:


print(clf.predict([[0, 1, 1]]))  # note: [0 1 1] not yet in the matrix


# In[5]:


# note: both example are in the matrix
print(clf.predict([[1, 0, 1], [0, 0, 1]]))
# Here, the output is [1 0], which is a vector of label.
# Each label is the prediction of he given classes


# In[6]:


dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph


# In the following cells, we start using the Iris dataset (from UCI Machine Learning repository)

# In[7]:


iris = load_iris()


# ## Type of prediction model and the working criteria
# 
# Declare the type of prediction model and the working criteria for the model induction algorithm
# 
# **Note**: from now on, we will use **two important parameters**:
# 
# 1. ```min_sample_spit```: used to control **overfitting** in **internal nodes**, 
# 2. ```min_sample_leaf```: used to control **overfitting** in **leaves**.
# 

# In[8]:


clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  random_state=300,
                                  min_samples_leaf=5,
                                  class_weight={0: 1, 1: 1, 2: 1})


# ## Split the dataset in training and test set

# In[9]:


# Generate a random permutation of the indices of examples that will be later
# used for the training and the test set
np.random.seed(0)

# randomly choose the indices of some entries to put in the training set.
# we use the randomly generated indices for training later.
indices = np.random.permutation(len(iris.data))

# We now decide to keep the last 10 indices for test set, the remaining
# for the training set
indices_training = indices[:-10]
indices_test = indices[-10:]

# keep for training all the matrix elements with the exception of the last 10
iris_X_train = iris.data[indices_training]
iris_y_train = iris.target[indices_training]

# keep the last 10 elements for test set
iris_X_test = iris.data[indices_test]
iris_y_test = iris.target[indices_test]


# ## Fit the learning model on training set

# In[10]:


# Fit the model to the training data
clf = clf.fit(iris_X_train, iris_y_train)


# ## Obtain predictions

# In[11]:


# Apply fitted model "clf" to the test set
predicted_y_test = clf.predict(iris_X_test)

# Print the predictions (class numbers associated to classes names
# in target names)
print("Predictions: \t{0}".format(predicted_y_test))
print("True classes:\t{0}".format(iris_y_test))
print("Iris target names: {0}".format({k: v for k, v in enumerate(iris.target_names,
                                                                  start=0)}))


# Print the index of the test instances and the corresponding predictions

# In[12]:


# Print the corresponding instances indexes and class names
for i in range(len(iris_y_test)):
    print("Instance # {0}:".format(indices_test[i]))
    print("Predicted: {0}\t True: {1}\n".format(iris.target_names[predicted_y_test[i]],
                                                iris.target_names[iris_y_test[i]]))


# ## Look at the specific examples

# In[13]:


for i in range(len(iris_y_test)): 
    print("Instance # {0} {1}:".format(i, indices_test))
    s = ""
    for j in range(len(iris.feature_names)):
        s = s + iris.feature_names[j] + "=" + str(iris_X_test[i][j])
        if j<len(iris.feature_names)-1: s = s + ", "
    print(s)
    print("Predicted: "+iris.target_names[predicted_y_test[i]]+"\t True: "+iris.target_names[iris_y_test[i]]+"\n")


# ## Obtain model performance results

# In[14]:


# Print some metrics results
acc_score = accuracy_score(iris_y_test, predicted_y_test)
print("Accuracy score: {0} = {0:.0%}".format(acc_score))
f1=f1_score(iris_y_test, predicted_y_test, average='macro')
print("F1 score: {0}".format(f1))


# ## Cross Validation and comparison with F1 Score

# In[15]:


# Cross_val_score will be used to separate training and test
iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=300, min_samples_leaf=5, class_weight={0:1,1:1,2:1})
clf = clf.fit(iris.data, iris.target)

scores = cross_val_score(clf, iris.data, iris.target, cv=5) # score will be the accuracy. cv=5 -> 5 folds
print("Cross-validation scores:\t{0}".format(scores))

# Computes F1-score
f1_scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print("F1 scores:\t\t\t{0}".format(f1_scores))


# ## Show the resulting tree 

# ### Print the picture in a PDF file

# In[16]:


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("./figures/my_iris_predictions")


# ### Generate a picture here

# In[17]:


print(list(iris.feature_names))
print(list(iris.target_names))


# In[18]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names, 
                         class_names=iris.target_names, 
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# ## Your work: what you have to do
# Modify the given Jupyter notebook on decision trees on Iris data and perform the following tasks:
# 
# 1. get an artificial inflation of some class in the training set by a given factor: 10 (weigh more the classes virginica e versicolor which are more difficult to discriminate). Learn the tree in these conditions.
# 2. modify the weight of some classes (set to 10 the weights for misclassification between virginica into versicolor and vice versa) and learn the tree in these conditions. You should obtain similar results as for step 
# 3. learn trees but try to avoid overfitting (by improving the error on the test set) tuning the hyper-parameters on: the minimum number of samples per leaf, max depth of the tree, min_impurity_decrease parameters, max leaf nodes, etc. Use misclassification error.
# 4. build the confusion matrix of the created tree models on the test set and show them. 
# 5. build the ROC curves (or coverage curves in coverage space) and plot them for each tree model you have created: for each model you have to build three curves, one for each class, considered in turn as the positive class. (1-vs-rest model)
# 
# for the last point, 2 possibility: 
# 
# 1. implement yourself the function which you need
# 2. or go in sklearn is there is thew empirical probability. Decision tree can be sed to compute classification probability of the class. In this case the teacher may ask you to explain which function you have used. (clf.predict_proba - Probability prediction foreach classifier)

# In[19]:


#help(tree._tree.Tree) # Help of sklearn.tree

