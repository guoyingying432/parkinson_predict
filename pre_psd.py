import numpy as np
import os
import csv
import myfunc

data_path1 = "data1/joints/"  # 骨架数据的位置
data_label1 = "data1/data1_label.csv"  # label文件的位置，label文件中的顺序和文件名一一对应
data_path2 = "data2/joints/"  # 骨架数据的位置
data_label2 = "data2/data2_label.csv"  # label文件的位置，label文件中的顺序和文件名一一对应
total_data1, total_label1 = myfunc.read_data(data_path1, data_label1)
total_data2, total_label2 = myfunc.read_data(data_path2, data_label2)

total_data = np.concatenate((total_data1,total_data2),axis=0)
total_label = np.concatenate((total_label1,total_label2),axis=0)

N = total_data.shape[0]

joints_psd_pre = np.zeros((N,2*299+1)) #用来存放需要去matlab里面处理的数据

for m in range(0,21):
    for k in range(0,N):
        length = (total_data[k, 0, :, m] == 0).argmax(axis=0)  #实际帧数
        if length==0:
            length=300
        joints_psd_pre[k,2*299]=length   #放在最后一列
        for i in range(0, length-1):   #计算x方向差
            dis_h = total_data[k, :, i + 1, :][0][m] - total_data[k, :, i, :][0][m]
            joints_psd_pre[k,i]=dis_h
        for i in range(0, length-1):   #计算y方向差
            dis_v = total_data[k, :, i + 1, :][1][m] - total_data[k, :, i, :][1][m]
            joints_psd_pre[k,i+299]=dis_v
        #是否要滤波，可以考虑
        #joints_psd_pre[k,0:299]=sgoal_filter(joints_psd_pre[k,0:299], 110, 11, 3)
        #joints_psd_pre[k,299:]=sgoal_filter(joints_psd_pre[k,299:], 110, 11, 3)

    path="pre_psd/alldata/"
    np.savetxt(path+str(m)+"joints_psd_pre.txt",joints_psd_pre,fmt="%d",delimiter=",")  #存成txt格式，用matlab去读
    print("第{}个关节pre_psd完成".format(m))
