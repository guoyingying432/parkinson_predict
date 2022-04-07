import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report#这个包是评价报告

data_svm=np.loadtxt('svm_data050.txt', delimiter=',')

feature=data_svm[:,0:data_svm.shape[1]-1]
rating=data_svm[:,data_svm.shape[1]-1]

X = feature
y = rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


#svr_rbf = SVR(kernel='rbf',C=1,gamma ='auto')
svr_lin = SVR(kernel='linear', C=1)
#svr_poly = SVR(kernel='poly', C=1, degree=2,gamma ='auto')

#y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
#y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

y_round=np.around(y_lin)
outcome=np.zeros((len(y_test),2))
outcome[:,0]=y_test
outcome[:,1]=y_round

r=0
for i in range(len(y_test)):
    if(y_round[i]==y_test[i]):
        r=r+1
acc=r/len(y_test)

print('done!')

print(classification_report(y_test, y_round))