#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from keras.layers import InputLayer, LSTM , Dense , Flatten


# In[2]:


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# In[3]:


df = pd.read_csv(csv_path)
df
df = df[5::6]
df


# In[4]:


df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df[:26]


# In[5]:


temp = df['T (degC)']
temp.plot()


# In[6]:


def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


# In[7]:


WINDOW_SIZE = 5
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape


# In[8]:


X_train1, y_train1 = X1[:60000], y1[:60000]
X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
X_test1, y_test1 = X1[65000:], y1[65000:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape


# In[12]:


model = Sequential()
model.add(InputLayer((5, 1)))
model.add(Flatten())
model.add(Dense(64, 'relu'))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))

model.summary()


# In[14]:


model.compile(loss='mse', optimizer='adam')


# In[15]:


history = model.fit(X_train1, y_train1, epochs=50, batch_size=64, validation_split=0.2, verbose=1)


# In[19]:


y_train_pred = model.predict(X_train1).flatten()
train_mse = mean_squared_error(y_train1, y_train_pred)
train_rmse = np.sqrt(train_mse)

y_test_pred = model.predict(X_test1).flatten()
test_mse = mean_squared_error(y_test1, y_test_pred)
test_rmse = np.sqrt(test_mse)


# In[20]:


print(f'Training MSE: {train_mse:.3f}')
print(f'Training RMSE: {train_rmse:.3f}')
print(f'Test MSE: {test_mse:.3f}')
print(f'Test RMSE: {test_rmse:.3f}')


# In[21]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(y_train1[:50], label='true')
ax1.plot(y_train_pred[:50], label='predicted')
ax1.set_title('Training set')
ax1.legend()

ax2.plot(y_test1[:50], label='true')
ax2.plot(y_test_pred[:50], label='predicted')
ax2.set_title('Test set')
ax2.legend()

plt.show()


# In[22]:


plt.figure(figsize=(8, 8))
plt.scatter(y_test1, y_test_pred)
plt.plot([0, 50], [0, 50], color='red')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()


# In[25]:


test_mae = mean_absolute_error(y_test1, y_test_pred)
test_mape = test_mae*100
train_mae = mean_absolute_error(y_train1, y_train_pred)
train_mape = train_mae*100

print("Test MAE:", test_mae)
print("Test MAPE:", test_mape)
print("Train MAE:", train_mae)
print("Train MAPE:", train_mape)


# In[24]:


train_corr, _ = pearsonr(y_train1, y_train_pred)
test_corr, _ = pearsonr(y_test1, y_test_pred)

print(f"Training set correlation: {train_corr:.3f}")
print(f"Test set correlation: {test_corr:.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




