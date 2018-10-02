# -*- coding: utf-8 -*-
"""
Keras tutorial

https://www.datacamp.com/community/tutorials/deep-learning-python
"""
import numpy as np
import pandas as pd

# Get the data, basic analysis
white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
#print "null reds: %s" % pd.isnull(red)
#pd.isnull(white)
red.info()
#print red.describe()
#print red.head()
#print red.tail()
#print red.sample(5)

import matplotlib.pyplot as plt
# Plot some data
fig, ax = plt.subplots(1, 2)
ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()

import seaborn as sns
# Combine into one dataset
red['type'] = 1
white['type'] = 0
wines = red.append(white, ignore_index=True)
corr = wines.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#sns.plt.show()

# Split the data up in train and test sets
from sklearn.model_selection import train_test_split
# Not sure about this indexing, was
# X=wines.ix[:,0:11]
X = wines.iloc[:,0:11]
y = np.ravel(wines.type)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

# Model the data
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))
# Add one hidden layer 
model.add(Dense(8, activation='relu'))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
# Model output shape
model.output_shape

# Model summary
print("model summary:")
model.summary()
# Model config
print("model config: ", model.get_config())
# List all weight tensors 
print("model weights: ", model.get_weights())

# Compile and fit
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

# Save the model
model_json = model.to_json()
with open('keras_tutorial_model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('keras_tutorial_model.hd5')

# Predict
y_pred = model.predict(X_test)
print("y_pred: %s" % y_pred[:5])
print("y_test: %s" % y_test[:5])

# Evaluate the model
score = model.evaluate(X_test, y_test,verbose=1)
print(score)