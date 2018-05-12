
# coding: utf-8

# In[81]:


import numpy as np


# In[82]:


import pandas as pd


# In[83]:


def load_data_and_label(data,labels):
    data_df = pd.read_csv(data,header=None)
    X = data_df.values.reshape(-1,28,28,4).astype(np.uint8)
    labels_df = pd.read_csv(labels,header=None)
    Y = labels_df.values.getfield(dtype=np.int8)
    return X,Y


# In[ ]:


x_train,y_train = load_data_and_label(data='deepsat-sat6/X_train_sat6.csv',labels='deepsat-sat6/y_train_sat6.csv')


# In[ ]:


x_test,y_test = load_data_and_label(data='deepsat-sat6/X_test_sat6.csv',labels='deepsat-sat6/y_test_sat6.csv')


# In[71]:


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# In[72]:


# Initialising the CNN
classifier = Sequential()


# In[73]:


# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 4), activation = 'relu'))


# In[74]:


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[75]:


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[76]:


# Step 3 - Flattening
classifier.add(Flatten())


# In[77]:


# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(units = 6, activation = 'softmax'))


# In[78]:


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[79]:


print(classifier.summary())


# In[80]:


# Part 2 - Fitting the CNN to the images



classifier.fit(x_train, y_train, batch_size=200, epochs=6, verbose=1, validation_data=(x_test, y_test))

