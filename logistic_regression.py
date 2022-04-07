import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告

data_svm=np.loadtxt('svm_data050.txt', delimiter=',')

feature=data_svm[:,0:data_svm.shape[1]-1] #feature是270x11的矩阵，每个病人11个特征
rating=data_svm[:,data_svm.shape[1]-1] #270个人对应的评分
rating=rating
feature_insert = np.insert(feature,0,1,axis=1)

lam=0.1  #学习率
theta=np.zeros(feature_insert.shape[1])

#sigmoid假设函数
def hypoth(theta,x):
    z = x.dot(theta)
    return 1/(1+np.exp(-z))
#代价函数
def costFunction(theta,x,y,lam):
    #未正则化部分
    unreg_first = (-y)*np.log(hypoth(theta,x)+1e-5)
    unreg_second = (1-y)*np.log(1-hypoth(theta,x)+1e-5)
    unreg = np.mean(unreg_first - unreg_second)
    #正则化部分（对theta0不进行正则化）
    reg = (lam/(2*x.shape[0])*np.sum(np.power(theta[1:],2)))
    return unreg + reg
#梯度函数
def gdReg(theta,x,y,lam):
    unreg = x.T.dot(hypoth(theta,x)-y)/x.shape[0]
    theta_0 = theta
    theta_0[0] = 0 #将theta0改为0，即对theta0不考虑正则化
    reg = (lam/x.shape[0])*theta_0
    return unreg + reg
#一对多分类
def onevsall(x,y,lam):
    y_lable=np.unique(y)
    numoftheta=x.shape[1]
    alltheta=np.zeros([len(y_lable),numoftheta])
    for i in range(len(y_lable)):
        theta=np.zeros(numoftheta)
        y_onevsall=np.array([1 if label == y_lable[i] else 0 for label in y] )
        fmin=opt.minimize(fun=costFunction,x0=theta,args=(x,y_onevsall,lam),method='TNC',jac=gdReg)
        alltheta[i,:]=fmin.x
    return alltheta
alltheta=onevsall(feature_insert,rating,lam)
#预测函数
def predict(theta,x):
    hy = hypoth(theta,x)
    p = np.argmax(hy,axis = 1) #argmax索引是从0开始的，而我们的是从1开始的
    return p

y_predict=predict(alltheta.T,feature_insert)
print(classification_report(rating,y_predict))