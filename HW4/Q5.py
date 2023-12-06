#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#(a)


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,precision_recall_fscore_support

# Load the dataset
data = pd.read_csv(r"C:\Users\smith\Desktop\NCSU-SEM-1\ALDA\HW\HW-4\svm_2023\data\svm_data_2023.csv")
data


# In[5]:


print(data["Class"].value_counts(normalize=True) *100)


# In[ ]:


#(b)


# In[11]:


X = data.drop(columns = 'Class')  # Features
y = data['Class']  # Labels

# Perform stratified random sampling to split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

print("Training data\n", y_train.value_counts(normalize=True) *100, "\n")
print("Testing data\n", y_test.value_counts(normalize=True) *100)


# In[ ]:


#(c)


# In[13]:


train_data = pd.read_csv(r"C:\Users\smith\Desktop\NCSU-SEM-1\ALDA\HW\HW-4\svm_2023\data\train_data_2023.csv")
test_data = pd.read_csv(r"C:\Users\smith\Desktop\NCSU-SEM-1\ALDA\HW\HW-4\svm_2023\data\test_data_2023.csv")

# Extract features and labels from the data
x_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]
y_train.value_counts()


# In[14]:


# Extract features and labels from the data
x_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]
y_test.value_counts()


# In[12]:


# Define the values of C to be tested
C_values = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]

# Initialize lists to store the number of support vectors
num_sv_list = []

# Loop through different values of C
for C in C_values:
    # Create an SVM classifier with a linear kernel and the current C value
    classifier = svm.SVC(C=C, kernel='linear')
    
    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)
    
    # Get the number of support vectors
    num_sv = np.sum(classifier.n_support_)
    num_sv_list.append(num_sv)
print("number of support vectors for different values of C: \n",num_sv_list)


# In[11]:


# Plot the results
plt.plot(C_values, num_sv_list, marker='o')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Number of Support Vectors')
plt.title('Number of Support Vectors vs. C')
plt.grid()
plt.show()


# In[ ]:


#(d)


# In[15]:


c_vals=[0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
degrees=[1, 2, 3, 4, 5]
coef0s=[0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
gammas=[0.0001,0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3]
output={}


# In[17]:


tuned_parameters ={"C": c_vals}
print("For Linear kernel")
grid = GridSearchCV(SVC(kernel="linear"), tuned_parameters, cv=5)
grid. fit(x_train,y_train)
print(" Linear kernel Best parameters are:")
best_model = grid.best_estimator_
print (grid. best_params_)
print()
grid_predictions = best_model. predict(x_test)
print("Linear Kernel Best Results are:")
print(classification_report(y_test,grid_predictions))
print()
accuracy = accuracy_score(y_test,grid_predictions)
precision, recall, f1score, support=precision_recall_fscore_support (y_test,grid_predictions, average="macro")
print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ",recall, "F1 score: ",f1score)
output["linear"]=[accuracy, precision, recall,f1score]


# In[20]:


tuned_parameters ={"C": c_vals,"degree": degrees,"coef0":coef0s}
print("For Polynomial kernel")
grid = GridSearchCV(SVC(kernel="poly"), tuned_parameters, cv=5)
grid. fit(x_train,y_train)
print(" Polynomial kernel Best parameters are:")
print (grid. best_params_)
best_model = grid.best_estimator_
print()
grid_predictions = best_model.predict(x_test)
print("Polynomial Kernel Best Results are:")
print(classification_report(y_test,grid_predictions))
print()
accuracy = accuracy_score(y_test,grid_predictions)
precision, recall, f1score, support=precision_recall_fscore_support (y_test,grid_predictions, average="macro")
print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ",recall, "F1 score: ",f1score)
output["poly"]=[accuracy, precision, recall,f1score]


# In[23]:


tuned_parameters ={"C": c_vals, "gamma": gammas}
print("For RBF kernel")
grid = GridSearchCV(SVC(kernel="rbf"), tuned_parameters, cv=5)
grid. fit(x_train,y_train)
print(" RBF kernel Best parameters are:")
print (grid. best_params_)
best_model = grid.best_estimator_
print()
grid_predictions = best_model.predict(x_test)
print("RBF Kernel Best Results are:")
print(classification_report(y_test,grid_predictions))
print()
accuracy = accuracy_score(y_test,grid_predictions)
precision, recall, f1score, support=precision_recall_fscore_support (y_test,grid_predictions, average="macro")
print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ",recall, "F1 score: ",f1score)
output["linear"]=[accuracy, precision, recall,f1score]


# In[24]:


tuned_parameters ={"C": c_vals, "gamma": gammas,"coef0":coef0s}
print("For Sigmoid kernel")
grid = GridSearchCV(SVC(kernel="sigmoid"), tuned_parameters, cv=5)
grid. fit(x_train,y_train)
print(" Sigmoid kernel Best parameters are:")
print (grid. best_params_)
best_model = grid.best_estimator_
print()
grid_predictions = best_model.predict(x_test)
print("Sigmoid Kernel Best Results are:")
print(classification_report(y_test,grid_predictions))
print()
accuracy = accuracy_score(y_test,grid_predictions)
precision, recall, f1score, support=precision_recall_fscore_support (y_test,grid_predictions, average="macro")
print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ",recall, "F1 score: ",f1score)
output["linear"]=[accuracy, precision, recall,f1score]


