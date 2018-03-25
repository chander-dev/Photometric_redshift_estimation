from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split
import pandas as pd 

path ='data_redshift.csv'
data =  pd.read_csv(path)

# put colors in a matrix

X = np.zeros((data.shape[0], 4))
X[:, 0] = data['modelMag_u'] - data['modelMag_g']
X[:, 1] = data['modelMag_g'] - data['modelMag_r']
X[:, 2] = data['modelMag_r'] - data['modelMag_i']
X[:, 3] = data['modelMag_i'] - data['modelMag_z']
z = np.asarray(data['z'])


X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size=0.2)

clf = DecisionTreeRegressor()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

axis_lim = np.array([0,0.4])

rms = np.sqrt(np.mean((Y_test - Y_pred) ** 2))
# print "RMS error = %.2g" % rms

ax = plt.axes()
plt.scatter(Y_test, Y_pred, c='k', lw=0, s=4)
plt.plot(axis_lim, axis_lim, '--k')
plt.plot(axis_lim, axis_lim + rms, ':r')
plt.plot(axis_lim, axis_lim - rms, ':r')
plt.xlim(axis_lim)
plt.ylim(axis_lim)

plt.text(0.99, 0.02, "RMS error = %.2g" % rms,
         ha='right', va='bottom', transform=ax.transAxes,
         bbox=dict(ec='w', fc='w'), fontsize=16)

plt.title('Photo-z: Decision Tree Regression',fontsize=20)
plt.xlabel(r'$\mathrm{z_{Pred}}$', fontsize=18)
plt.ylabel(r'$\mathrm{z_{True}}$', fontsize=18)
plt.show()