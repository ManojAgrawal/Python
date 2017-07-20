
# coding: utf-8

# This code is based on excellent kernel by Peter Grenholm on Kaggle https://www.kaggle.com/toregil/welcome-to-deep-learning-cnn-99. I have made some modifications on the original.

# In[16]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
get_ipython().magic('matplotlib inline')


# In[17]:

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# In[18]:

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[19]:

train_df.head()


# In[20]:

train_df.info()


# Check the distribution of labels

# In[21]:

train_df['label'].value_counts()


# Split into features and labels

# In[22]:

X = train_df.iloc[:,:-1].copy()

y = train_df.iloc[:,-1].copy()
 


# Lets check one of the images

# In[23]:

img = np.array(X.iloc[1,:].values.reshape(28,28))


# In[24]:

plt.imshow(img,cmap='gray')


# I am making the feature set binary as either there is a value in the pixel or there isn't. This should make all the images uniformly dark.

# In[25]:

X[X > 0] = 1
img = np.array(X.iloc[1,:].values.reshape(28,28))
plt.imshow(img,cmap='gray')


# Split the training set further into training and test sets

# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# Convolution Networks work on arrays so we need to convert dataframes into arrays and reshape each row into 28 X 28 

# In[27]:

X_train_arr = np.array(X_train)
X_test_arr = np.array(X_test)


# In[28]:

X_train = X_train_arr.reshape(-1,28,28,1)
X_test = X_test_arr.reshape(-1,28,28,1)


# One hot encoding for labels

# In[31]:

y_train = to_categorical(y_train)
y_test =  to_categorical(y_test)
print(y_train[0])


# Train the model using Sequential API

# In[41]:

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[42]:

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)


# Compile the model. Use logloss function which is called categorical cross entropy in Keras.

# In[43]:

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])


# We train once with a smaller learning rate to ensure convergence. We then speed things up, only to reduce the learning rate by 10% every epoch. Keras has a function for this:

# In[44]:

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)


# Fit the model

# In[49]:

hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                           steps_per_epoch= 2000,
                           epochs= 200, 
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(X_test, y_test), #For speed
                           callbacks=[annealer])


# Evaluate the model on the test (hold out) set

# In[50]:

final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


# Final loss: 0.0210, final accuracy: 0.9949

# In[51]:

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()


# In[52]:

y_hat = model.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)


# In[53]:

test_df[test_df > 0] = 1
test = np.array(test_df).reshape(-1,28,28,1)


# In[54]:

test_prob = model.predict(test, batch_size=64)
test_pred = np.argmax(test_prob,axis=1)


# In[55]:

np.savetxt("C:/Users/Manoj/python scripts/digits/digits_submission2.csv",test_pred,delimiter=',')


# In[ ]:



