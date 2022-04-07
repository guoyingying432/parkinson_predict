import myfunc
import numpy as np
import scipy.io
import math

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

for m in range(0,21):

    joint=np.zeros((N,24))  #每一个关节有一个特征矩阵，纵坐标是270个病人，横坐标是23个特征加一个等级评分

    #运动学特征和凸包，共16个特征
    for k in range(0,N):     #k是人数
        # displacement有关的
        dis_list = []
        length=(total_data[k,0,:,m]==0).argmax(axis=0)  #实际帧数
        if length==0:
            length=300
        for i in range(0, length-1):
            x1 = total_data[k, :, i, :][0][m]
            y1 = -total_data[k, :, i, :][1][m]
            x2 = total_data[k, :, i + 1, :][0][m]
            y2 = -total_data[k, :, i + 1, :][1][m]
            dis = math.sqrt(abs(int(x1) - int(x2)) ** 2 + abs(int(y1) - int(y2)) ** 2)
            dis_list.append(dis)
        dis = []
        dis = myfunc.sgoal_filter(dis_list, 110,11,3)
        Q1 = np.percentile(dis, 25)
        Q3 = np.percentile(dis, 75)
        IQRspd = Q3 - Q1
        joint[k,0]=max(dis)
        joint[k,1]=np.median(dis)
        joint[k,2]=np.mean(dis)
        joint[k,3]=np.std(dis)
        joint[k,4]=IQRspd

        #acc有关的
        acc_list = []
        for i in range(0, length-2):
            x1 = dis_list[i]
            y1 = dis_list[i + 1]
            acc = y1 - x1
            acc_list.append(acc)
        acc = []
        acc = myfunc.sgoal_filter(acc_list, 110,11,3)
        Q1 = np.percentile(acc, 25)
        Q3 = np.percentile(acc, 75)
        IQRacc = Q3 - Q1
        joint[k,5]=max(acc)
        joint[k,6]=np.median(acc)
        joint[k,7]=np.mean(acc)
        joint[k,8]=np.std(acc)
        joint[k,9]=IQRacc

        jerk_list = []
        for i in range(0, length-3):
            x1 = acc_list[i]
            y1 = acc_list[i + 1]
            jerk = y1 - x1
            jerk_list.append(jerk)
        jerk = []
        jerk = myfunc.sgoal_filter(jerk_list, 110,11,3)
        Q1 = np.percentile(jerk, 25)
        Q3 = np.percentile(jerk, 75)
        IQRjerk = Q3 - Q1
        joint[k,10]=max(jerk)
        joint[k,11]=np.median(jerk)
        joint[k,12]=np.mean(jerk)
        joint[k,13]=np.std(jerk)
        joint[k,14]=IQRjerk

        #凸包
        P = []
        for i in range(0, length):
            x_con = total_data[k, :, i, :][0][m]
            y_con = -total_data[k, :, i, :][1][m]
            joint1 = (x_con, y_con)
            P.append(joint1)
        L = myfunc.GrahamScan(P)
        P = np.array(P)
        L = list(reversed(L))
        joint[k, 15] = myfunc.GetAreaOfPolyGonbyVector(L)

        #print("运动学和凸包特征,complete {}/270".format(k+1))

    #谱域7个特征
    path1="post_psd/alldata/"
    data=scipy.io.loadmat(path1+str(m)+'joints.mat')
    psd=data['joints01_psd']
    joint[:,16:23]=psd[:,:]  #添加到特征矩阵里

    #等级
    joint[:,23]=total_label

    joint[np.isnan(joint)]=0  #nan转变成0
    print("第{}个关节特征提取完成".format(m))

    path="spearman/alldata/"
    np.savetxt(path+str(m)+"joint_spearman.txt",joint,delimiter=",")