import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
plt.ion()

path ='data_redshift.csv'
data =  pd.read_csv(path)

mag = np.vstack([data['modelMag_%s' % f] for f in 'ugriz']).T
z = data['z']

X_train, X_test, Y_train, Y_test = train_test_split(mag, z, test_size=0.2)

def compute_photoz_forest(depth):
    rms_test = np.zeros(len(depth))
    rms_train = np.zeros(len(depth))
    i_best = 0
    z_fit_best = None

    for i, d in enumerate(depth):
        clf = RandomForestRegressor(n_estimators=10,
                                    max_depth=d, random_state=0)
        clf.fit(X_train, Y_train)

        X_fit_train = clf.predict(X_train)
        X_fit = clf.predict(X_test)
        rms_train[i] = np.sqrt(np.mean((X_fit_train - Y_train) ** 2))
        rms_test[i] = np.sqrt(np.mean((X_fit - Y_test) ** 2))

        if rms_test[i] <= rms_test[i_best]:
            i_best = i
            X_fit_best = X_fit

    return rms_test, rms_train, i_best, X_fit_best


depth = np.arange(1, 21)
rms_test, rms_train, i_best, X_fit_best = compute_photoz_forest(depth)
best_depth = depth[i_best]

#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(wspace=0.25,
                    left=0.1, right=0.95,
                    bottom=0.15, top=0.9)

# left panel: plot cross-validation results
ax = fig.add_subplot(121)
ax.plot(depth, rms_test, '-k', label='cross-validation')
ax.plot(depth, rms_train, '--k', label='training set')
ax.legend(loc=1)

ax.set_xlabel('Number of depth')
ax.set_ylabel('rms error')

ax = fig.add_subplot(122)
ax.scatter(Y_test, X_fit_best, s=1, lw=0, c='k')
ax.plot(axis_lim, axis_lim, '--k')
ax.plot(axis_lim, axis_lim + rms, ':r')
ax.plot(axis_lim, axis_lim - rms, ':r')
ax.xlim(axis_lim)
ax.ylim(axis_lim)

ax.plot([0, 0.4], [0, 0.4], ':k')
ax.text(0.03, 0.97, "depth = %i\nrms = %.3f" % (best_depth, rms_test[i_best]),
        ha='left', va='top', transform=ax.transAxes)

ax.set_xlabel(r'$z_{\rm true}$',fontsize=18)
ax.set_ylabel(r'$z_{\rm fit}$',fontsize=18)
plt.suptitle('Photo-z:Random Forest',fontsize=20)
plt.show()