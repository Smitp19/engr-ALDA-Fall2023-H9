import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

X = np.array([[10.81,0.0],[6.61,5.0],[2.57,1.0],[9.48,5.0],[0.2,8.0],[2.11,2.0],[6.7,1.0],[-1.75,5.0],[2.54,-4.0],[0.81,-6.0]])
Y = np.array([0,1,1,0,1,1,0,1,0,0])

#a
loo = LeaveOneOut()
knn = KNeighborsClassifier(n_neighbors=1)

train_set_X,train_set_Y,actual_output,expected_output=[],[],[],[]

for i, (train_index, test_index) in enumerate(loo.split(X)):
    for index in train_index:
        train_set_X.append(np.array(X[index]))
        train_set_Y.append(Y[index])


    knn.fit(train_set_X,train_set_Y)
    knn.predict(X[test_index])
    expected_output.append(Y[test_index[0]])
    output=knn.predict(np.array(X[test_index]))
    actual_output.append(output[0])
    train_set_X.clear()
    train_set_Y.clear()

print("Leave one out cross validation error: ",mean_squared_error(expected_output,actual_output))

#b
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
print("3 Nearest neighbours of data point 3:",knn.kneighbors([X[2]]))
print("3 Nearest neighbours of data point 5:",knn.kneighbors([X[4]]))

#c
train_sets_X, test_sets_X,train_sets_Y,test_sets_Y=[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]
# 3-fold distribution based on given condition
knn = KNeighborsClassifier(n_neighbors=3)
for i in range(len(X)):
        test_sets_X[i%3].append(X[i])
        test_sets_Y[i%3].append(Y[i])
        
        train_sets_X[(i+2)%3].append(X[i])
        train_sets_Y[(i+2)%3].append(Y[i])
        
        train_sets_X[(i+1)%3].append(X[i])
        train_sets_Y[(i+1)%3].append(Y[i])
predicted_Y=[[],[],[]]
for i in range(len(train_sets_X)):
    knn.fit(train_sets_X[i],train_sets_Y[i])
    predicted_Y[i]=np.array(knn.predict(test_sets_X[i]))

error=0
for i in range(len(train_sets_Y)):
    error += mean_squared_error(test_sets_Y[i],predicted_Y[i])
print("3-folded cross validation error of 3NN for the given data set: ", error/3)