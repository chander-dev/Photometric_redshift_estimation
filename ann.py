import keras 
import pandas as pd 
import numpy as np 
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.callbacks import Callback
plt.ion()

path ='/home/chander/Documents/study_material/data_science/project/photometric_redshift/data_redshift.csv'
data =  pd.read_csv(path)

mag = np.vstack([data['modelMag_%s' % f] for f in 'ugriz']).T
z = data['z']
rate=0.2

#20 % for testing and 10 % for validation and 70 % for training 
X_train, X_test, Y_train, Y_test = train_test_split(mag, z, test_size=0.2)

model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate))
model.add(BatchNormalization())
model.add(Dense(16,activation='relu'))
model.add(Dropout(rate))
model.add(BatchNormalization())
model.add(Dense(4, activation='relu'))
model.add(Dropout(rate))
model.add(BatchNormalization())
model.add(Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam')
# history= model.fit(X_train,Y_train,nb_epoch=10,shuffle=True,validation_split=0.1)
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=10)
X_fit_train = model.predict(X_train)
X_fit = model.predict(X_test)
rms_train = np.mean(np.sqrt((X_fit_train.squeeze() - Y_train) ** 2))
rms_test = np.mean(np.sqrt((X_fit.squeeze() - Y_test) ** 2))


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()