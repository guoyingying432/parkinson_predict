import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report#这个包是评价报告

data_svm=np.loadtxt('svm_data050.txt', delimiter=',')
feature=data_svm[:,0:data_svm.shape[1]-1] #feature是270x11的矩阵，每个病人11个特征
rating=data_svm[:,data_svm.shape[1]-1] #270个人对应的评分
rating_reshape=np.zeros((len(rating),1))
rating_reshape[:,0]=rating[:]

enc=OneHotEncoder(sparse=False)
y_enc=enc.fit_transform(rating_reshape)

#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#前向传播
def fwd_prop(x,theta1,theta2):
    a1=np.insert(x,0,1,axis=1)  #a1要insert
    z2=np.dot(a1,theta1.T)
    a2=sigmoid(z2)
    a2_insert=np.insert(a2,0,1,axis=1) #a2要insert
    z3=np.dot(a2_insert,theta2.T)
    a3=sigmoid(z3) #不用insert
    return a1,z2,a2_insert,z3,a3
def costfunc(theta1,theta2,inputsize,hiddensize,numlabels,x,y,rate):
    a1, z2, a2, z3, a3=fwd_prop(x,theta1,theta2)
    m=x.shape[0]

    J=0
    first=np.multiply(-y,np.log(a3))  #点乘
    second=np.multiply(-(1-y),np.log(1-a3))
    J=np.sum(np.sum(first+second,axis=1))/m
    return J

#初始化 这是一些网络设置
inputsize=18
hiddensize=10
numlabels=5
rate=1

#带正则化的代价函数
def costfuncreg(theta1,theta2,inputsize,hiddensize,numlabels,x,y,rate):
    m=x.shape[0]
    J=costfunc(theta1,theta2,inputsize,hiddensize,numlabels,x,y,rate)
    J+=(rate/(2*m))*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
    return J

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))
#np.random.random
params=(np.random.random(size=hiddensize*(inputsize+1)+numlabels*(hiddensize+1))-0.5)*0.24

#反向传播
def backprop(params,inputsize,hiddensize,numlabels,x,y,rate):
    m=x.shape[0]
    theta1=np.reshape(params[0:hiddensize*(inputsize+1)],(hiddensize,(inputsize+1)))
    theta2=np.reshape(params[hiddensize*(inputsize+1):],(numlabels,(hiddensize+1)))
    a1, z2, a2, z3, a3 = fwd_prop(x, theta1, theta2)
    J=0
    #大三角
    delta1=np.zeros(theta1.shape)
    delta2=np.zeros(theta2.shape)
    J=costfuncreg(theta1,theta2,inputsize,hiddensize,numlabels,x,y,rate)
    #backprop
    for i in range(m):
        a1i=np.mat(a1[i,:])
        z2i=np.mat(z2[i,:])
        a2i=np.mat(a2[i,:])
        z3i=np.mat(z3[i,:])
        a3i=np.mat(a3[i,:])
        yi=np.mat(y[i,:])

        d3i=a3i-yi   #(1,10)
        z2i_insert=np.insert(z2i,0,1)
        #(1,26)
        d2i=np.multiply(np.dot(d3i,theta2),sigmoid_gradient(z2i_insert))

        delta1+=np.dot((d2i[:,1:]).T,a1i)
        delta2+=np.dot(d3i.T,a2i)
    #正则化 注意j=0时
    gra1=delta1/m+rate*theta1
    gra1[:,1:]=delta1[:,1:]/m
    gra2=delta2/m+rate*theta2
    gra2[:,1:]=delta2[:,1:]/m

    grad=np.concatenate((gra1.ravel(),gra2.ravel())) #里面要加一个括号

    return J,grad

#args用来初始化fun，x0是要求的
fmin = minimize(fun=backprop, x0=(params), args=(inputsize, hiddensize, numlabels, feature, y_enc, rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)

thetafinal1=np.reshape(fmin.x[:hiddensize*(inputsize+1)],(hiddensize,inputsize+1))
thetafinal2=np.reshape(fmin.x[hiddensize*(inputsize+1):],(numlabels,hiddensize+1))
a1, z2, a2, z3, h = fwd_prop(feature, thetafinal1, thetafinal2 )
y_predict = np.array(np.argmax(h, axis=1) )

print(classification_report(rating, y_predict))