import numpy as np
import os
import csv
import myfunc

#读取数据
data_path1 = "data1/joints/"  # 骨架数据的位置
data_label1 = "data1/data1_label.csv"  # label文件的位置，label文件中的顺序和文件名一一对应
data_path2 = "data2/joints/"  # 骨架数据的位置
data_label2 = "data2/data2_label.csv"  # label文件的位置，label文件中的顺序和文件名一一对应
total_data1, total_label1 = myfunc.read_data(data_path1, data_label1)
total_data2, total_label2 = myfunc.read_data(data_path2, data_label2)

total_data = np.concatenate((total_data1,total_data2),axis=0)
total_label = np.concatenate((total_label1,total_label2),axis=0)
N = total_data.shape[0]

cor=np.loadtxt('spearman_cor.txt', delimiter=',')

feature=(['max speed','median speed','mean speed','std speed','IQR speed',
          'max acc','median acc','mean acc','std acc','IQR acc',
          'max jerk','median jerk','mean jerk','std jerk','IQR jerk',
          'disp peak','disp entropy','disp tot power','disp 0.5-1','disp >2','disp >4','disp >6',
          'conv hull'])

fearture_index=np.argwhere(np.fabs(cor)>0.5) #找出绝对值大于0.5的

print('相关性>0.5的有：')
for i in range(0,fearture_index.shape[0]):
    print('第'+str(fearture_index[i,0])+'个关节的'+feature[fearture_index[i,1]])


#生成用于存放svm数据的矩阵，纵坐标是744个病人，横坐标是11个特征和一个等级
data_svm=np.zeros((N,fearture_index.shape[0]+1))
for i in range(0,fearture_index.shape[0]):
    data=np.loadtxt('spearman/alldata/'+str(fearture_index[i,0])+'joint_spearman.txt', delimiter=',')
    data_svm[:,i]=data[:,fearture_index[i,1]]
data_svm[:,fearture_index.shape[0]]=total_label

print('用于svm的数据提取完成')
np.savetxt("svm_data050.txt",data_svm,delimiter=",")


