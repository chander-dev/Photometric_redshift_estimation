import keras 
import pandas as pd 
import numpy as np 
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.callbacks import Callback
from keras import backend as K
from sklearn import preprocessing
plt.ion()

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

path ='data_redshift.csv'
data =  pd.read_csv(path)

mag = np.vstack([data['modelMag_%s' % f] for f in 'ugriz']).T
z = data['z']
rate=0.2

scaler = preprocessing.StandardScaler()
#20 % for testing and 10 % for validation and 70 % for training 
X_train, X_test, Y_train, Y_test = train_test_split(mag, z, test_size=0.2)
# X_train = scaler.fit_transform(X_train)
# mu = scaler.mean_
# var = scaler.var_

model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(rate))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss=root_mean_squared_error, optimizer='adam')
# history= model.fit(X_train,Y_train,nb_epoch=10,shuffle=True,validation_split=0.1)
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=3)
# X_fit_train = model.predict(X_train)
# rms_train = np.mean(np.sqrt((X_fit_train.squeeze() - Y_train) ** 2))

# X_test= (X_test- mu)/var

X_fit = model.predict(X_test)

rms_test = np.sqrt(np.mean((X_fit.flatten() - Y_test)**2))


# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
ax = plt.axes()

plt.text(0.99, 0.02, "RMS error = %.2g" % rms_test,
         ha='right', va='bottom', transform=ax.transAxes,
         bbox=dict(ec='w', fc='w'), fontsize=16)

plt.title('ANN Prediction',fontsize=20)
plt.xlabel(r'$\mathrm{z_{Pred}}$', fontsize=18)
plt.ylabel(r'$\mathrm{z_{True}}$', fontsize=18)
plt.show()