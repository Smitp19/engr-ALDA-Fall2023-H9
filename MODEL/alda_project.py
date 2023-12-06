# -*- coding: utf-8 -*-
"""ALDA PROJECT.ipynb

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

df.duplicated().sum()
df.drop_duplicates()
print("17 duplicate values ")
df['date_time']=pd.to_datetime(df['date_time'])
df['day'] = df['date_time'].dt.day_name()
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['hour'] = df['date_time'].dt.hour
df.drop('date_time',axis=1,inplace = True)
from tabulate import tabulate



#Print descriptive statistics
table = (df['traffic_volume'].describe()).to_frame()
print(tabulate(table, headers= ['statistic', 'values'], tablefmt='psql', numalign="right"))

## Generate an histogram graph
plt.figure(figsize=(10,4))
plt.subplot(1,1,1)
with sns.axes_style("darkgrid"):
    plt.hist(df["traffic_volume"],color="#1984c5")
    plt.xlabel('Traffic Volume (cars/hr)', labelpad=10, fontsize='12')
    plt.ylabel('Frequency', labelpad=10, fontsize='12')
    plt.title('Westbound Traffic Distribution on I-94', loc='left', fontstyle='italic', pad=15, fontsize=14, fontweight='bold')
    plt.grid(False)
    plt.xlim(-100,7500)
    plt.ylim(0,8500)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tick_params(direction='out')
    sns.despine()
    plt.axhline(y = 0, color = 'black', linestyle = '-', linewidth=2)
    plt.show()

catcol = ['holiday','weather_main','weather_description','day']
encoder =  LabelEncoder()
for col in catcol:
    df[col] = encoder.fit_transform(df[col])

df['weather_description'].unique()

x = df.drop('traffic_volume',axis = 1)
y = df['traffic_volume']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.80,shuffle = True,random_state=50)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score


# Initialize the Linear Regression model
lr = LinearRegression()

# Train the model on the training data
lr.fit(x_train, y_train)

# Make predictions
pred_train = lr.predict(x_train)
pred_test = lr.predict(x_test)

print('Training Accuracy:')
print('mean_absolute_error:', mean_absolute_error(y_train, pred_train))
print('mean_squared_error:', mean_squared_error(y_train, pred_train))

print('Testing Accuracy:')
print('mean_absolute_error:', mean_absolute_error(y_test, pred_test))
print('mean_squared_error:', mean_squared_error(y_test, pred_test))

# Calculate R-squared for training data
r2_train = r2_score(y_train, pred_train)

# Calculate R-squared for testing data
r2_test = r2_score(y_test, pred_test)

print('Training R-squared:', r2_train)
print('Testing R-squared:', r2_test)

# Initialize the Random Forest model
rm = RandomForestRegressor()

# Train the model on the training data
rm.fit(x_train, y_train)

# Make predictions
pred_train = rm.predict(x_train)
pred_test = rm.predict(x_test)

print('Training Accuracy:')
print('mean_absolute_error:', mean_absolute_error(y_train, pred_train))
print('mean_squared_error:', mean_squared_error(y_train, pred_train))

print('Testing Accuracy:')
print('mean_absolute_error:', mean_absolute_error(y_test, pred_test))
print('mean_squared_error:', mean_squared_error(y_test, pred_test))

# Calculate R-squared for training data
r2_train = r2_score(y_train, pred_train)

# Calculate R-squared for testing data
r2_test = r2_score(y_test, pred_test)

print('Training R-squared:', r2_train)
print('Testing R-squared:', r2_test)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the KNN model
knn = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors as needed

# Train the model on the training data
knn.fit(x_train, y_train)

# Make predictions
knn_pred_train = knn.predict(x_train)
knn_pred_test = knn.predict(x_test)

# Evaluate the model using MAE, MSE, RMSE, and R-squared
mae_train = mean_absolute_error(y_train, knn_pred_train)
mae_test = mean_absolute_error(y_test, knn_pred_test)
mse_train = mean_squared_error(y_train, knn_pred_train)
mse_test = mean_squared_error(y_test, knn_pred_test)
rmse_train = mse_train ** 0.5
rmse_test = mse_test ** 0.5
r2_train = r2_score(y_train, knn_pred_train)
r2_test = r2_score(y_test, knn_pred_test)

print('Training Accuracy:')
print('Mean Absolute Error:', mae_train)
print('Mean Squared Error:', mse_train)
print('Root Mean Squared Error:', rmse_train)
print('R-squared:', r2_train)

print('Testing Accuracy:')
print('Mean Absolute Error:', mae_test)
print('Mean Squared Error:', mse_test)
print('Root Mean Squared Error:', rmse_test)
print('R-squared:', r2_test)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
selected = ['holiday', 'temp', 'clouds_all', 'weather_description']
Y = df['traffic_volume']
X = df['traffic_volume'].to_numpy()
X.shape
not_scaled = X.reshape(X.shape[0],1)
not_scaled.shape
# Scale the data (important for LSTM)
scaler = MinMaxScaler()
traffic_volume = X.reshape(-1, 1)
scaled_close = scaler.fit_transform(traffic_volume )
scaled_close.shape

import tensorflow as tf
scaled_close_tf = tf.constant(scaled_close, dtype=tf.float32)
not_scaled_tf = tf.constant(not_scaled, dtype=tf.float32)

"""## Preprocessing"""

SEQ_LEN = 6

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.8)

X_train.shape

y_train.shape

from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import  Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed
from tensorflow import keras

"""## Building Bi-LSTM Model"""

DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

model = keras.Sequential()

#input layer
model.add(Bidirectional(LSTM(128*3, return_sequences=True, input_shape=(WINDOW_SIZE, X_train.shape[-1]))))
model.add(Dropout(rate=0.2))

#hidden layer
model.add(Bidirectional(LSTM((128 * 2), return_sequences=True)))
model.add(Dropout(rate=0.2))

model.add(Bidirectional(LSTM((128 * 2), return_sequences=True)))
model.add(Dropout(rate=0.2))

model.add(Bidirectional(LSTM((128), return_sequences=False)))
model.add(Dropout(rate=0.2))


# model.add(Dense(units=512))
# model.add(Dropout(rate=0.2))

#output layer
model.add(Dense(units=1))

"""## Training model

"""

# defining learning rate scheduler function
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

m = tf.keras.metrics.RootMeanSquaredError()
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=[m]
)

BATCH_SIZE = 64

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[callback],
    verbose=1
)

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

y_pred

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
# Evaluate the model using MAE, MSE, RMSE, and R-squared

mae_test = mean_absolute_error(y_test, y_pred)

mse_test = mean_squared_error(scaler.inverse_transform(y_test).ravel(), scaler.inverse_transform(y_pred).ravel())
rmse_train = mse_train ** 0.5
rmse_test = mse_test ** 0.5

r2_test = r2_score(y_test, y_pred)


print('Testing Accuracy:')
print('Mean Absolute Error:', mae_test)
print('Mean Squared Error:', mse_test)
print('Root Mean Squared Error:', rmse_test)
print('R-squared:', r2_test)

from xgboost import XGBRegressor

# Initialize the XGB
xgb = XGBRegressor()


# Train the model on the training data
xgb.fit(x_train, y_train)

# Make predictions
pred_train = xgb.predict(x_train)
pred_test = xgb.predict(x_test)

print('Training Accuracy:')
print('mean_absolute_error:', mean_absolute_error(y_train, pred_train))
print('mean_squared_error:', mean_squared_error(y_train, pred_train))

print('Testing Accuracy:')
print('mean_absolute_error:', mean_absolute_error(y_test, pred_test))
print('mean_squared_error:', mean_squared_error(y_test, pred_test))

# Calculate R-squared for training data
r2_train = r2_score(y_train, pred_train)

# Calculate R-squared for testing data
r2_test = r2_score(y_test, pred_test)

print('Training R-squared:', r2_train)
print('Testing R-squared:', r2_test)

models = {
    'LinearRegression':LinearRegression(),
    'KNN': KNeighborsRegressor(n_neighbors=5),  # You can adjust n_neighbors as needed
    'RandomForest' : RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LSTM' : keras.Sequential()
}


data = {}
model_list = []
train_r2_score = []
test_r2_score = []
for i in range(len(models)):
    model = list(models.values())[i]
    model.fit(x_train,y_train)

    # Prediction
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    model_name = list(models.keys())[i]
    model_list.append(model_name)
    print('--------------------------------------------------------------------')
    print()
    print('Model Name :',model_name)
    print('Training Accuracy: ')
    print('mean_absolute_error: ',mean_absolute_error(y_train,pred_train))
    print('mean_squared_error: ',mean_squared_error(y_train,pred_train))
    print('r2_score: ',r2_score(y_train,pred_train))
    print()
    print('Testing Accuracy: ')

    print('mean_absolute_error: ',mean_absolute_error(y_test,pred_test))
    print('mean_squared_error: ',mean_squared_error(y_test,pred_test))
    print('r2_score: ',r2_score(y_test,pred_test))
    print()

    train_r2_score.append(r2_score(y_train,pred_train))
    test_r2_score.append(r2_score(y_test,pred_test))

data['model'] = model_list
data['train_r2_score'] = train_r2_score
data['test_r2_score'] = test_r2_score

data = pd.DataFrame(data)
data

sns.barplot(y='model',x = 'test_r2_score',data=data,orient='h')
