import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv('hr_data.csv')
#print df.head()

y = df.left
X = df.ix[:,('satisfaction_level','last_evaluation','average_montly_hours')].values

#X = sm.add_constant(X)
#print X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#print X_train.shape, y_train.shape
#print X_test.shape, y_test.shape

'''Linear regression'''
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print(predictions)[0:5]
print (lm.score(X_test, y_test))


#est = sm.OLS(y_train, X_train).fit()
#print est.summary()

#X_prime = np.linspace(X.satisfaction_level.min(),X.satisfaction_level.max(), 100)[:,np.newaxis]
#X_prime = sm.add_constant(X_prime)

#X_test = sm.add_constant(X_test)

#y_hat = est.predict(X_test)
#print (est.summary())

#print np.mean((y_hat - y_test)**2)
#print "accuracy:{:.4f".format(est.score(X_test, y_test))
#plt.scatter(X,y,alpha=0.3)
#plt.xlabel("Satisfaction Level")
#plt.ylabel("Left")
#plt.plot(X_test[:,1],y_hat,'r',alpha=0.9)
#plt.show()
